from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import Counter
import re
import string
from Datasets import Custom_Dataset


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        # Model Initializaion
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, cache_dir=hparams.cache_dir, truncation_side='left', use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            hparams.model_name_or_path, dropout=0, attention_dropout=0, activation_dropout=0, cache_dir=hparams.cache_dir)

        self.save_hyperparameters(hparams)
        self.model.resize_token_embeddings(len(self.tokenizer))
        # getting the index of the target set if there is multiple val sets
        self.target_validation_idx = None

    def on_fit_end(self):
        if self.hparams.save_checkpoint:
            self.model.save_pretrained(
                f'checkpoints/{self.hparams.wandb_run_name}')
            self.tokenizer.save_pretrained(
                f'checkpoints/{self.hparams.wandb_run_name}')
        return super().on_fit_end()

    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels
        )
        loss, score = outputs[0], outputs[1]
        return loss, score

    def training_step(self, batch, batch_idx):
        loss, score = self._step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.hparams.loss_fn == 'negative':
            return loss * -1
        elif self.haprams.loss_fn == 'mle':
            return loss  # standard MLE
        else:
            raise Exception(
                f'{self.haparams.loss_fn} is not a vliad loss function')

    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        return self.validation_general_lm(batch)

    # Measures benchmark tasks
    def validation_general_lm(self, batch):
        task = batch["task"][0]
        task_type = batch["task_type"][0]

        if task_type == 'ppl':
            loss, score = self._step(batch)
            self.log(
                f'{task}/loss',
                loss,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                add_dataloader_idx=False,
                sync_dist=True)
        elif task_type == 'classification':
            self.classification_verbalizer(
                padding_length=self.hparams.input_length,
                task=task,
                batch=batch,
                choices=batch["choices"],
                answer_index=batch["answer_index"])
        elif task_type == 'dialog':
            self.dialog_evaluation(
                padding_length=self.hparams.input_length,
                task=task,
                batch=batch)
        else:
            raise Exception(f'Currently, {task_type} not implemented..')

    def classification_verbalizer(
            self, padding_length, task, batch, choices, answer_index):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"]
        batch_size = len(source_ids)
        answer_idx = [-1] * batch_size
        for i in range(batch_size):
            answer_idx[i] = answer_index[i]

        batch_acc = 0

        inps = []
        cont_toks_list = []
        inplens = []

        answers = torch.zeros(batch_size, len(choices), device=self.device)

        for c_idx in range(len(choices)):
            choice_ids = self.tokenizer.batch_encode_plus(
                list(
                    choices[c_idx]),
                max_length=self.hparams.input_length,
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                return_tensors="pt")["input_ids"].tolist()
            for i in range(batch_size):
                context_enc = self.get_rid_of_pad(source_ids[i])
                continuation_enc = self.get_rid_of_pad(choice_ids[i])

                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(padding_length):],
                    dtype=torch.long
                ).to(self.device)
                inplen, = inp.shape
                cont = continuation_enc

                # pad length from seq to padding_length
                inp = torch.cat([
                    inp,  # [seq]
                    # [padding_length - seq]
                    torch.zeros(padding_length - inplen,
                                dtype=torch.long).to(inp.device) + self.tokenizer.pad_token_id
                ], dim=0)
                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
            multi_logits = F.log_softmax(self._model_call(
                batched_inps), dim=-1)  # [batch, padding_length, vocab]
            cnt = 0
            for logits, inp, inplen, cont_toks \
                    in zip(multi_logits, inps, inplens, cont_toks_list):

                # Slice to original seq length
                contlen = len(cont_toks)
                original_logits = logits

                # [1, seq, vocab]
                logits = logits[inplen - contlen - 1:inplen - 1].unsqueeze(0)
                # Check if per-token argmax is exactly equal to continuation
                cont_toks = torch.tensor(
                    cont_toks, dtype=torch.long).unsqueeze(0).to(
                    self.device)  # [1, seq]

                logits = torch.gather(
                    logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                # Answer: (log prob, is-exact-match)
                loss = -float(logits.sum())
                answers[cnt][c_idx] = loss
                cnt += 1
            inps = []
            cont_toks_list = []
            inplens = []

        answer_idx = torch.Tensor(answer_idx).to(self.device)
        answers = torch.argmin(answers, dim=1)

        batch_acc = int(torch.where(answers == answer_idx, 1, 0).sum())

        batch_acc_avg = batch_acc / batch_size

        self.log(
            f'{task}/acc' if '/' not in task else f'{task}_acc',
            batch_acc_avg,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)

        return

    def dialog_evaluation(self, padding_length, task, batch):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"].tolist()
        batch_size = len(source_ids)

        inps, cont_toks_list, inplens = [], [], []
        for i in range(batch_size):
            context_enc = self.get_rid_of_pad(source_ids[i])
            continuation_enc = self.get_rid_of_pad(target_ids[i])

            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(padding_length):],
                dtype=torch.long
            ).to(self.device)
            inplen, = inp.shape
            cont = continuation_enc

            # pad length from seq to padding_length
            inp = torch.cat([
                inp,  # [seq]
                # [padding_length - seq]
                torch.zeros(padding_length - inplen,
                            dtype=torch.long).to(inp.device) + self.tokenizer.pad_token_id
            ], dim=0)
            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
        # [batch, padding_length, vocab]
        multi_logits = self._model_call(batched_inps)

        full_logits, full_cont_toks = [], []
        for logits, inp, inplen, cont_toks \
                in zip(multi_logits, inps, inplens, cont_toks_list):

            # Slice to original seq length
            contlen = len(cont_toks)

            if contlen >= padding_length:
                cont_toks = cont_toks[:int(padding_length / 2)]
                contlen = len(cont_toks)

            # [seq, vocab]
            logits = logits[inplen - contlen - 1:inplen - 1]
            # Check if per-token argmax is exactly equal to continuation
            cont_toks = torch.tensor(
                cont_toks, dtype=torch.long).to(self.device)  # [seq]

            assert logits.shape[0] == cont_toks.shape[0]

            full_logits.append(logits)
            full_cont_toks.append(cont_toks)

        full_logits = torch.cat(full_logits)
        full_cont_toks = torch.cat(full_cont_toks)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(full_logits, full_cont_toks)

        generate_input = []
        for source_id in source_ids:
            inplen = len(source_id)
            inp = torch.tensor(source_id, dtype=torch.long).to(self.device)
            inp = torch.cat([
                torch.zeros(padding_length - inplen,
                            dtype=torch.long).to(inp.device) + self.tokenizer.pad_token_id,
                inp
            ], dim=0)
            generate_input.append(inp.unsqueeze(0))  # [1, padding_length]

        inputs = torch.cat(generate_input, dim=0)
        attention_masks = inputs.ne(self.tokenizer.pad_token_id).long()
        generated_ids = self.model.generate(
            inputs, attention_mask=attention_masks, max_new_tokens=32)[:, padding_length:]
        generated_text = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True)
        generated_text = [t.split('\nUser ')[0] for t in generated_text]
        target_text = self.tokenizer.batch_decode(
            target_ids, skip_special_tokens=True)

        # Debugging
        # source_text = self.tokenizer.batch_decode(source_ids, skip_special_tokens=True)
        # for s, g, t in zip(source_text, generated_text, target_text):
        #     print('---------------------')
        #     print(f'S: {s}')
        #     print(f'G: {g}')
        #     print(f'T: {t}')
        #     print('---------------------')

        f1_batched = 0
        for g, t in zip(generated_text, target_text):
            f1_batched += self._f1_score(g, t)

        unigram_f1 = f1_batched / batch_size

        self.log(
            f'{task}/loss' if '/' not in task else f'{task}_loss',
            loss,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True),
        self.log(
            f'{task}/f1' if '/' not in task else f'{task}_f1',
            unigram_f1,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)

    def configure_optimizers(self):
        parameters = self.model.parameters()
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98))
        return [optimizer]

    def get_dataset(self, dataset_name, tokenizer,
                    valid_subset_path, type_path):
        dataset = Custom_Dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            valid_subset_path=valid_subset_path,
            type_path=type_path,
            input_length=self.hparams.input_length,
            output_length=self.hparams.output_length,
            args=self.hparams)
        return dataset

    def train_dataloader(self):
        dataset = self.hparams.train_set
        train_dataset = self.get_dataset(
            dataset_name=dataset,
            tokenizer=self.tokenizer,
            valid_subset_path="",
            type_path="train")

        dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        datasets = []
        target_idx = -1
        for i in range(len(self.hparams.valid_sets)):
            dataset = self.hparams.valid_sets[i]
            valid_subset_path = self.hparams.valid_subset_path[i]
            type_path = self.hparams.valid_type_path[i]
            dataset_name = dataset

            dataset = self.get_dataset(
                dataset_name=dataset_name,
                tokenizer=self.tokenizer,
                valid_subset_path=valid_subset_path,
                type_path=type_path)
            datasets.append(dataset)

        dataloaders = []
        for i, dataset in enumerate(datasets):
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.hparams.eval_batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False))
        return dataloaders

    # Below are some utils functions

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            res = self.model(inps)
            return res[0][:, :, :]

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text

        def replace_space(text):
            return text.replace(u'\xa0', u' ')

        def remove_dialog_prompts(text):
            text = text.replace('user 1', '')
            text = text.replace('user 2', '')
            return text

        s = lower(s)
        s = remove_punc(s)
        s = remove_articles(s)
        # s = remove_dialog_prompts(s)
        s = replace_space(s)
        s = white_space_fix(s)
        return s

    def get_rid_of_pad(self, tokens):
        while tokens[-1] == -100 or tokens[-1] == self.tokenizer.pad_token_id:
            tokens.pop()
        return tokens

    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @property
    def max_length(self):
        return 2048

    @property
    def device(self):
        return self._device

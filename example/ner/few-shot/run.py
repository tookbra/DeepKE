import os

import hydra

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from hydra import utils
from torch.utils.data import DataLoader
from deepke.name_entity_re.few_shot.models.model import PromptBartModel, PromptGeneratorModel
from deepke.name_entity_re.few_shot.module.datasets import ConllNERProcessor, ConllNERDataset
from deepke.name_entity_re.few_shot.module.train import Trainer
from deepke.name_entity_re.few_shot.module.metrics import Seq2SeqSpanMetric
from deepke.name_entity_re.few_shot.utils.util import get_loss, set_seed
from deepke.name_entity_re.few_shot.module.mapping_type import mit_movie_mapping, mit_restaurant_mapping, atis_mapping

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import wandb

writer = wandb.init(project="DeepKE_NER_Few")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_CLASS = {
    'conll2003': ConllNERDataset,
    'mit-movie': ConllNERDataset,
    'mit-restaurant': ConllNERDataset,
    'atis': ConllNERDataset
}

DATA_PROCESS = {
    'conll2003': ConllNERProcessor,
    'mit-movie': ConllNERProcessor,
    'mit-restaurant': ConllNERProcessor,
    'atis': ConllNERProcessor
}

DATA_PATH = {
    'conll2003': {'train': 'data/conll2003/train.txt',
                  'dev': 'data/conll2003/dev.txt',
                  'test': 'data/conll2003/test.txt'},
    'mit-movie': {'train': 'data/mit-movie/20-shot-train.txt',
                  'dev': 'data/mit-movie/test.txt'},
    'mit-restaurant': {'train': 'data/mit-restaurant/10-shot-train.txt',
                       'dev': 'data/mit-restaurant/test.txt'},
    'atis': {'train': 'data/atis/20-shot-train.txt',
             'dev': 'data/atis/test.txt'}
}

MAPPING = {
    'conll2003': {'loc': '<<location>>',
                  'per': '<<person>>',
                  'org': '<<organization>>',
                  'misc': '<<others>>'},
    'mit-movie': mit_movie_mapping,
    'mit-restaurant': mit_restaurant_mapping,
    'atis': atis_mapping
}


@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    print(cfg)

    data_path = DATA_PATH[cfg.train.dataset_name]
    for mode, path in data_path.items():
        data_path[mode] = os.path.join(cfg.cwd, path)
    dataset_class, data_process = DATASET_CLASS[cfg.train.dataset_name], DATA_PROCESS[cfg.train.dataset_name]
    mapping = MAPPING[cfg.train.dataset_name]

    set_seed(cfg.train.seed)  # set seed, default is 1
    if cfg.train.save_path is not None:  # make save_path dir
        cfg.train.save_path = os.path.join(cfg.train.save_path, cfg.train.dataset_name + "_" + str(cfg.train.batch_size) + "_" + str(
            cfg.train.learning_rate) + cfg.train.notes)
        if not os.path.exists(cfg.train.save_path):
            os.makedirs(cfg.train.save_path, exist_ok=True)

    process = data_process(data_path=data_path, mapping=mapping, bart_name=cfg.train.bart_name,
                           learn_weights=cfg.train.learn_weights)
    train_dataset = dataset_class(data_processor=process, mode='train')
    train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=cfg.train.batch_size,
                                  num_workers=4)

    dev_dataset = dataset_class(data_processor=process, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, collate_fn=dev_dataset.collate_fn, batch_size=cfg.train.batch_size,
                                num_workers=4)

    label_ids = list(process.mapping2id.values())

    prompt_model = PromptBartModel(tokenizer=process.tokenizer, label_ids=label_ids, args=cfg)
    model = PromptGeneratorModel(prompt_model=prompt_model, bos_token_id=0,
                                 eos_token_id=1,
                                 max_length=cfg.train.tgt_max_len, max_len_a=cfg.train.src_seq_ratio, num_beams=cfg.train.num_beams,
                                 do_sample=False,
                                 repetition_penalty=1, length_penalty=cfg.train.length_penalty, pad_token_id=1,
                                 restricter=None)
    metrics = Seq2SeqSpanMetric(eos_token_id=1, num_labels=len(label_ids), target_type='word')
    loss = get_loss

    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=None, model=model, args=cfg,
                      logger=logger, loss=loss,
                      metrics=metrics, writer=writer)
    trainer.train()


if __name__ == "__main__":
    main()

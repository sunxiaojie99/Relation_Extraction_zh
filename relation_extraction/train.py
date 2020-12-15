import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

from .data_utils import WikiData, get_idx2label, load_checkpoint, save_checkpoint
from .model import SentenceRE
from .my_metrics import score

here = os.path.dirname(os.path.abspath(__file__))


def train(hparams):
    device = hparams.device
    seed = hparams.seed
    # 设置 (CPU) 生成随机数的种子，使得在每次重新运行程序时，同样的随机数生成代码得到的是同样的结果。
    torch.manual_seed(seed)
    # 可以让每次重新从头训练网络时的权重的初始值虽然是随机生成的但却是固定的

    pretrained_model_path = hparams.pretrained_model_path
    train_file = hparams.train_file
    validation_file = hparams.validation_file
    log_dir = hparams.log_dir
    output_dir = hparams.output_dir
    label_set_file = hparams.label_set_file
    checkpoint_file = hparams.checkpoint_file
    additional_tokens_file = hparams.additional_tokens_file
    model_file = hparams.model_file
    complete_model_file = hparams.complete_model_file
    not_need_label_file = hparams.not_need_label_file
    checkpoint_macro_file = hparams.checkpoint_macro_file
    checkpoint_micro_file = hparams.checkpoint_micro_file
    special_type_file = hparams.special_type_file

    max_len = hparams.max_len
    train_batch_size = hparams.train_batch_size
    validation_batch_size = hparams.validation_batch_size
    epochs = hparams.epochs

    learning_rate = hparams.learning_rate
    weight_decay = hparams.weight_decay
    is_add_entity_type = hparams.is_add_entity_type

    # train_dataset
    train_dataset = WikiData(data_file_path=train_file, not_need_label_file=not_need_label_file,
                             labels_path=label_set_file, additional_file_path=additional_tokens_file,
                             pretrained_model_path=pretrained_model_path, max_len=max_len,
                             is_add_entity_type=is_add_entity_type, special_type_file=special_type_file)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True)

    # model
    idx2label = get_idx2label(label_set_file)
    hparams.label_set_size = len(idx2label)
    model = SentenceRE(hparams).to(device)

    # load checkpoint if one exists
    if os.path.exists(checkpoint_file):
        checkpoint_dict = load_checkpoint(checkpoint_file)
        checkpoint_macro_dict = load_checkpoint(checkpoint_macro_file)
        checkpoint_micro_dict = load_checkpoint(checkpoint_micro_file)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1  # 默认从最好的之后再开始训
        model.load_state_dict(torch.load(model_file))  # 把之前保存的权重加载到现在的模型中
    else:
        checkpoint_dict = {}
        checkpoint_macro_dict = {}
        checkpoint_micro_dict = {}
        best_f1 = 0.0
        epoch_offset = 0

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device=device)
    running_loss = 0.0
    # 定义log的保存路径
    writer = SummaryWriter(os.path.join(
        log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))

    for epoch in range(epoch_offset, epochs):  # 之前训练到的epoch，到我们要求的epoch总数
        print("Epoch: {}".format(epoch))
        model.train()
        for i_batch, sampled_batched in enumerate(tqdm(train_loader, desc='Training')):
            token_ids = sampled_batched['token_ids'].to(device)
            token_type_ids = sampled_batched['token_type_ids'].to(device)
            attention_mask = sampled_batched['attention_mask'].to(device)
            e1_mask = sampled_batched['e1_mask'].to(device)
            e2_mask = sampled_batched['e2_mask'].to(device)
            label_ids = sampled_batched['label_id'].to(device)
            model.zero_grad()  # 每一个batch重新累计梯度
            # print(token_ids[0], token_type_ids[0], attention_mask[0], e1_mask[0])
            logits = model(token_ids, token_type_ids,
                           attention_mask, e1_mask, e2_mask)  # 和forward对应
            loss = criterion(logits, label_ids)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

            if i_batch % 10 == 9:  # 9, 19, 29, .. 每10个batch输出一次平均loss
                writer.add_scalar('Training/training loss', scalar_value=running_loss / 10,
                                  global_step=epoch * len(train_loader) + i_batch)
                running_loss = 0.0

        # 每个epoch都对验证集计算一次loss
        if validation_file:
            validation_dataset = WikiData(data_file_path=validation_file, not_need_label_file=not_need_label_file,
                                          labels_path=label_set_file, additional_file_path=additional_tokens_file,
                                          pretrained_model_path=pretrained_model_path, max_len=max_len,
                                          is_add_entity_type=is_add_entity_type, special_type_file=special_type_file)
            val_loader = DataLoader(
                validation_dataset, batch_size=validation_batch_size, shuffle=True)
            model.eval()
            with torch.no_grad():
                labels_true = []
                labels_pred = []
                for i, sampled_batched in enumerate(tqdm(val_loader, desc='Validation')):
                    token_ids = sampled_batched['token_ids'].to(device)
                    token_type_ids = sampled_batched['token_type_ids'].to(
                        device)
                    attention_mask = sampled_batched['attention_mask'].to(
                        device)
                    e1_mask = sampled_batched['e1_mask'].to(device)
                    e2_mask = sampled_batched['e2_mask'].to(device)
                    label_ids = sampled_batched['label_id'].to(device)
                    logits = model(token_ids, token_type_ids,
                                   attention_mask, e1_mask, e2_mask)  # 和forward对应
                    # [batch_size, label_set_size]
                    pred_tag_ids = logits.argmax(1)
                    labels_true.extend(label_ids.tolist())  # 把tensor转为list
                    labels_pred.extend(pred_tag_ids.tolist())

                localtime = time.asctime(time.localtime(time.time()))
                print("本地时间为 :", localtime)

                print(metrics.classification_report(labels_true, labels_pred, labels=list(idx2label.keys()),
                                                    target_names=list(idx2label.values())))

                weighted_f1 = metrics.f1_score(
                    labels_true, labels_pred, average='weighted')
                weighted_precision = metrics.precision_score(
                    labels_true, labels_pred, average='weighted')
                weighted_recall = metrics.recall_score(
                    labels_true, labels_pred, average='weighted')
                accuracy = metrics.accuracy_score(labels_true, labels_pred)
                macro_f1 = metrics.f1_score(
                    labels_true, labels_pred, average='macro')
                micro_precision_c, micro_recall_c, micro_f1_c = score(
                    labels_true, labels_pred)

                writer.add_scalar('Validation/micro_f1_c', micro_f1_c, epoch)
                writer.add_scalar('Validation/weighted-f1', weighted_f1, epoch)
                writer.add_scalar(
                    'Validation/weighted-precision', weighted_precision, epoch)
                writer.add_scalar('Validation/weighted-recall',
                                  weighted_recall, epoch)
                writer.add_scalar('Validation/accuracy', accuracy, epoch)
                writer.add_scalar('Validation/macro_f1', macro_f1, epoch)

                if checkpoint_dict.get('epoch_f1'):
                    checkpoint_dict['epoch_f1'][epoch] = weighted_f1
                    checkpoint_macro_dict['epoch_macro_f1'][epoch] = macro_f1
                    checkpoint_micro_dict['epoch_micro_f1'][epoch] = micro_f1_c

                else:
                    checkpoint_dict['epoch_f1'] = {epoch: weighted_f1}
                    checkpoint_macro_dict['epoch_macro_f1'] = {epoch: macro_f1}
                    checkpoint_micro_dict['epoch_micro_f1'] = {
                        epoch: micro_f1_c}

                if weighted_f1 > best_f1:
                    best_f1 = weighted_f1
                    checkpoint_dict['best_f1'] = weighted_f1
                    checkpoint_dict['best_epoch'] = epoch
                    torch.save(model.state_dict(), model_file)  # 只是参数
                    torch.save(model, complete_model_file)  # 保存完整的模型，方便调用

                # 每个epoch都存一下
                save_checkpoint(checkpoint_dict, checkpoint_file)
                save_checkpoint(checkpoint_macro_dict, checkpoint_macro_file)
                save_checkpoint(checkpoint_micro_dict, checkpoint_micro_file)

    writer.close()

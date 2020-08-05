#機器安裝
#pip3 install https://github.com/huggingface/pytorch-transformers/archive/1.2.0.zip
#rm -rf bert-chinese-qa*
#wget -q --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1GQtGFd-1AvZHZuYckhA3xqvvpDk-x5DW' -O bert-chinese-qa.zip
#unzip bert-chinese-qa.zip -d bert-chinese-qa

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)


def to_list(tensor):
    return tensor.detach().cpu().tolist()
 

def _get_best_indexes(logits, n_best_size=1):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes
 

def evaluate(dataset, model, tokenizer):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    all_results = []
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      }
            example_indices = batch[3]
            outputs = model(**inputs)
            start_logits = to_list(outputs[0][0])
            end_logits   = to_list(outputs[1][0])
            start_indexes = _get_best_indexes(start_logits)
            end_indexes = _get_best_indexes(end_logits)
    return (start_indexes, end_indexes)

import collections

from torch.utils.data import TensorDataset




def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examples_to_features(tokenizer, question_text, doc_tokens, max_seq_length=512,
                                 doc_stride=128, max_query_length=64,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""
    query_tokens = tokenizer.tokenize(question_text)
    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3


    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        
        p_mask = []

        
        if not cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = 0

        
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

        
        tokens.append(sep_token)
        segment_ids.append(sequence_a_segment_id)
        p_mask.append(1)

        
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(0)
        paragraph_len = doc_span.length

        
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        p_mask.append(1)

        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = len(tokens) - 1  

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)
            p_mask.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_mask = torch.tensor([input_mask], dtype=torch.long)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long)
    cls_index = torch.tensor([cls_index], dtype=torch.long)
    p_mask = torch.tensor([p_mask], dtype=torch.float)
    example_index = torch.arange(input_ids.size(0), dtype=torch.long)
    data = TensorDataset(input_ids, input_mask, segment_ids,
                            example_index, cls_index, p_mask)



    return data, tokens

# 因為要我們今天要跑的是中文QA 所以只有Bert可以用
import torch
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer)


device = torch.device("cuda")

checkpoint = 'bert-chinese-qa'
config_class, model_class, tokenizer_class = BertConfig, BertForQuestionAnswering, BertTokenizer
model = model_class.from_pretrained(checkpoint).to(device)
tokenizer = tokenizer_class.from_pretrained('bert-base-chinese', do_lower_case=True)

#記得上傳json檔
import json

path_Q='http://fgcgame.stpi.narl.org.tw/json/ FE9043C9-E1BB-4C2F-921E-DD06FD6118E0/305C7DBE-7CBF-490C-9E8E-937A8C1E3927/question.json '
#path_A='FGC_release_A_answers.json'

#檔案編碼方式：UTF-8
with open(path_Q, 'r',encoding="utf-8") as reader:
    QA_Q = json.loads(reader.read())

#with open(path_A, 'r',encoding="utf-8") as reader:
#    QA_A = json.loads(reader.read())

Q_tolo=0
A_bad=0
tolo=len(QA_Q)
for i in range(tolo):
#for i in range(3):
  essay=QA_Q[i]['DTEXT']#原文章
  
  #長度修正
  #抓取需要修正多少
  #抓題目的tokensize


  for j in range(len(QA_Q[i]['QUESTIONS'])):#該大題有幾小題
    Q_tolo+=1
    question=QA_Q[i]['QUESTIONS'][j]['QTEXT']#問題
    QID=QA_Q[i]['QUESTIONS'][j]['QID']#題號


    t=len(tokenizer.tokenize(QA_Q[i]['QUESTIONS'][j]['QTEXT']))+4

    #前500    
    context=essay[0:512-t]
    #Bert 作答
    data, tokens = convert_examples_to_features(tokenizer=tokenizer, question_text=question, doc_tokens=context)
    start, end = evaluate(data, model, tokenizer)
    A1="".join(tokens[start[0]: end[0]+1])#答案

    #後500    
    context=essay[0+t:]
    #Bert 作答
    data, tokens = convert_examples_to_features(tokenizer=tokenizer, question_text=question, doc_tokens=context)
    start, end = evaluate(data, model, tokenizer)
    A2="".join(tokens[start[0]: end[0]+1])#答案


    if(A1=='[CLS]'):
      if(A2=='[CLS]'):
        AA='####不知道####'
        A_bad+=1
      else:
        AA=A2
    else:
      if(A2=='[CLS]'):
        AA=A1
      else:#兩邊都有答案但答案不同 (選前500))
        AA=A1

    #回傳格式
    print('305C7DBE-7CBF-490C-9E8E-937A8C1E3927','FE9043C9-E1BB-4C2F-921E-DD06FD6118E0',QID,AA,sep='\t', end='\n')

#print('答的出來的比率:',Q_tolo-A_bad,'/',Q_tolo,'\t',float(Q_tolo-A_bad)/float(Q_tolo)*100,'%')



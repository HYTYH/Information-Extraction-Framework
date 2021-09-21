import os
import re
import sys
import argparse
import matplotlib.pyplot as plt
import networkx as nx

from cnsenti import Emotion
from cnsenti import Sentiment
from naiveKGQA import NaiveKGQA
from harvesttext import HarvestText
from harvesttext.resources import get_qh_sent_dict

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ht = HarvestText()
senti = Sentiment()
emotion = Emotion()

os_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
filedir = os.path.join(os_path, '信息获取')
print(filedir)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true', default=False, help='Generate all infos')
parser.add_argument('--basic', action='store_true', default=False, help='Extract basic infos')
parser.add_argument('--ent', action='store_true', default=False, help='Recognize entities')
parser.add_argument('--k', type=int, default=5, help='topK keywords')
parser.add_argument('--doc', type=int, default=100, help='Num of docs to extract infos')
parser.add_argument('--specificdoc', type=str, default="", help='Specific doc to extract')
parser.add_argument('--sentence', action='store_true', default=False, help='analyze syntax of sentences')
parser.add_argument('--sents', action='store_true', default=False, help='analyze sentiment of passage')
parser.add_argument('--event', action='store_true', default=False, help='Distillation of event')
parser.add_argument('--para', action='store_true', default=False, help='Cut paragraph')
parser.add_argument('--p', type=int, default=10, help='Goal number of paragraphs')
parser.add_argument('--qa', action='store_true', default=False, help='Raise questions to system')
parser.add_argument('--network', action='store_true', default=False, help='Generate a social network')
parser.add_argument('--q', type=str, default='', help='When enabled qa mode, this is use to record questions')
args ,_= parser.parse_known_args()

punc = ['\n', '；', '.', '、', '（', '）', '！', '@', '#', '，', '￥', '%', '…',
        '&', '*', '-', '+', '~', '·', '|', '【', '】', '：', '“', '？', '《', '》', '。', '\'']

# Measure
tp_title = fp_title = fn_title = tn_title = 0
tp_dept = fp_dept = fn_dept = tn_dept = 0
tp_time = fp_time = fn_time = tn_time = 0
tp_count = fp_count = fn_count = tn_count = 0
tp_school = fp_school = fn_school = tn_school = 0
tp_kwds = fp_kwds = fn_kwds = tn_kwds = 0
tp_nwds = fp_nwds = fn_nwds = tn_nwds = 0

def read(passage):
    fo = open(f'数据/{passage}.txt', "r+", encoding='utf-8')

    passage = fo.read()
    fo.close()

    return passage

def CalcResult(fo):
    StandardOutput('Result', fo)

    global tp_title, fp_title, fn_title, tn_title
    global tp_dept, fp_dept, fn_dept, tn_dept
    global tp_time, fp_time, fn_time, tn_time
    global tp_count, fp_count, fn_count, tn_count
    global tp_school, fp_school, fn_school, tn_school
    global tp_kwds, fp_kwds, fn_kwds, tn_kwds
    global tp_nwds, fp_nwds, fn_nwds, tn_nwds

    if tp_title + tn_title + fp_title + fn_title == 0:
        acc_title = 1
    else:
        acc_title = (tp_title + tn_title) / (tp_title + tn_title + fp_title + fn_title)
    if tp_title + fp_title == 0:
        precision_title = 1
    else:
        precision_title = tp_title / (tp_title + fp_title)
    if tp_title + fn_title == 0:
        recall_title = 1
    else:
        recall_title = tp_title / (tp_title + fn_title)
    if precision_title + recall_title == 0:
        f1_score_title = 1
    else:
        f1_score_title = 2 * precision_title * recall_title / (precision_title + recall_title)

    if tp_dept + tn_dept + fp_dept + fn_dept == 0:
        acc_dept = 1
    else:
        acc_dept = (tp_dept + tn_dept) / (tp_dept + tn_dept + fp_dept + fn_dept)
    if tp_dept + fp_dept == 0:
        precision_dept = 1
    else:
        precision_dept = tp_dept / (tp_dept + fp_dept)
    if tp_dept + fn_dept == 0:
        recall_dept = 1
    else:
        recall_dept = tp_dept / (tp_dept + fn_dept)
    if precision_dept + recall_dept == 0:
        f1_score_dept = 1
    else:
        f1_score_dept = 2 * precision_dept * recall_dept / (precision_dept + recall_dept)

    if tp_time + tn_time + fp_time + fn_time == 0:
        acc_time = 1
    else:
        acc_time = (tp_time + tn_time) / (tp_time + tn_time + fp_time + fn_time)
    if tp_time + fp_time == 0:
        precision_time = 1
    else:
        precision_time = tp_time / (tp_time + fp_time)
    if tp_time + fn_time == 0:
        recall_time = 1
    else:
        recall_time = tp_time / (tp_time + fn_time)
    if precision_time + recall_time == 0:
        f1_score_time = 1
    else:
        f1_score_time = 2 * precision_time * recall_time / (precision_time + recall_time)

    if tp_count + tn_count + fp_count + fn_count == 0:
        acc_count = 1
    else:
        acc_count = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)
    if tp_count + fp_count == 0:
        precision_count = 1
    else:
        precision_count = tp_count / (tp_count + fp_count)
    if tp_count + fn_count == 0:
        recall_count = 1
    else:
        recall_count = tp_count / (tp_count + fn_count)
    if precision_count + recall_count == 0:
        f1_score_count = 1
    else:
        f1_score_count = 2 * precision_count * recall_count / (precision_count + recall_count)

    if tp_school + tn_school + fp_school + fn_school == 0:
        acc_school = 1
    else:
        acc_school = (tp_school + tn_school) / (tp_school + tn_school + fp_school + fn_school)
    if tp_school + fp_school == 0:
        precision_school = 1
    else:
        precision_school = tp_school / (tp_school + fp_school)
    if tp_school + fn_school == 0:
        recall_school = 1
    else:
        recall_school = tp_school / (tp_school + fn_school)
    if precision_school + recall_school == 0:
        f1_score_school = 1
    else:
        f1_score_school = 2 * precision_school * recall_school / (precision_school + recall_school)

    if tp_kwds + tn_kwds + fp_kwds + fn_kwds == 0:
        acc_kwds = 1
    else:
        acc_kwds = (tp_kwds + tn_kwds) / (tp_kwds + tn_kwds + fp_kwds + fn_kwds)
    if tp_kwds + fp_kwds == 0:
        precision_kwds = 1
    else:
        precision_kwds = tp_kwds / (tp_kwds + fp_kwds)
    if tp_kwds + fn_kwds == 0:
        recall_kwds = 1
    else:
        recall_kwds = tp_kwds / (tp_kwds + fn_kwds)
    if precision_kwds + recall_kwds == 0:
        f1_score_kwds = 1
    else:
        f1_score_kwds = 2 * precision_kwds * recall_kwds / (precision_kwds + recall_kwds)

    if tp_nwds + tn_nwds + fp_nwds + fn_nwds == 0:
        acc_nwds = 1
    else:
        acc_nwds = (tp_nwds + tn_nwds) / (tp_nwds + tn_nwds + fp_nwds + fn_nwds)
    if tp_nwds + fp_nwds == 0:
        precision_nwds = 1
    else:
        precision_nwds = tp_nwds / (tp_nwds + fp_nwds)
    if tp_nwds + fn_nwds == 0:
        recall_nwds = 1
    else:
        recall_nwds = tp_nwds / (tp_nwds + fn_nwds)
    if precision_nwds + recall_nwds == 0:
        f1_score_nwds = 1
    else:
        f1_score_nwds = 2 * precision_nwds * recall_nwds / (precision_nwds + recall_nwds)

    print(f'           {"标题":>9} {"部门":>9} {"时间":>9} {"浏览量":>9.5} {"学院":>9} {"关键字":>8} {"相关词":>8}')
    print(f'Accuracy : {int(acc_title*100):>10}% {int(acc_dept*100):>10}% {int(acc_time*100):>10}% {int(acc_count*100):>10}% {int(acc_school*100):>10}% {int(acc_kwds*100):>10}% {int(acc_nwds*100):>10}%')
    print(f'Precision: {int(precision_title*100):>10}% {int(precision_dept*100):>10}% {int(precision_time*100):>10}% {int(precision_count*100):>10}% {int(precision_school*100):>10}% {int(precision_kwds*100):>10}% {int(precision_nwds*100):>10}%')
    print(f'Recall   : {int(recall_title*100):>10}% {int(recall_dept*100):>10}% {int(recall_time*100):>10}% {int(recall_count*100):>10}% {int(recall_school*100):>10}% {int(recall_kwds*100):>10}% {int(recall_nwds*100):>10}%')
    print(f'F1-score : {int(f1_score_title*100):>10}% {int(f1_score_dept*100):>10}% {int(f1_score_time*100):>10}% {int(f1_score_count*100):>10}% {int(f1_score_school*100):>10}% {int(f1_score_kwds*100):>10}% {int(f1_score_nwds*100):>10}%')

    fo.write(f'           {"标题":>9} {"部门":>9} {"时间":>9} {"浏览量":>9.5} {"学院":>9} {"关键字":>8} {"相关词":>8}\n')
    fo.write(f'Accuracy : {int(acc_title * 100):>10}% {int(acc_dept * 100):>10}% {int(acc_time * 100):>10}% {int(acc_count * 100):>10}% {int(acc_school * 100):>10}% {int(acc_kwds * 100):>10}% {int(acc_nwds * 100):>10}%\n')
    fo.write(f'Precision: {int(precision_title * 100):>10}% {int(precision_dept * 100):>10}% {int(precision_time * 100):>10}% {int(precision_count * 100):>10}% {int(precision_school * 100):>10}% {int(precision_kwds * 100):>10}% {int(precision_nwds * 100):>10}%\n')
    fo.write(f'Recall   : {int(recall_title * 100):>10}% {int(recall_dept * 100):>10}% {int(recall_time * 100):>10}% {int(recall_count * 100):>10}% {int(recall_school * 100):>10}% {int(recall_kwds * 100):>10}% {int(recall_nwds * 100):>10}%\n')
    fo.write(f'F1-score : {int(f1_score_title * 100):>10}% {int(f1_score_dept * 100):>10}% {int(f1_score_time * 100):>10}% {int(f1_score_count * 100):>10}% {int(f1_score_school * 100):>10}% {int(f1_score_kwds * 100):>10}% {int(f1_score_nwds * 100):>10}%\n')


def StandardOutput(type, fo):
    if type == 'Basic':
        out = "文章基本信息提取结果"
    elif type == 'Entity':
        out = "文章实体提取结果"
    elif type == 'Event':
        out = "文章事件提取结果 (三元组表示)"
    elif type == 'Sentence':
        out = "文章词语语法关系提取结果"
    elif type == 'Sents':
        out = "文章情感分析结果"
    elif type == 'Cut':
        out = "根据文章内容智能分段提取结果"
    elif type == 'Question':
        out = "针对提问系统给出的回答 (实验功能)"
    elif type == 'Result':
        out = '文章信息抽取指标'

    if type == 'Network':
        out1 = "基于关键字的关系网络提取 (可视化)"
        out2 = "可视化图片存储在./提取结果/关系网络/ 目录下"
        outBounder = "*" * max(len(out1), len(out2)) * 2
        print(f'\n\n{outBounder}')
        print(f'{outBounder}')
        print(f'{out1}')
        print(f'{out2}')
        print(f'{outBounder}')
        print(f'{outBounder}\n')
        fo.write(f'\n\n{outBounder}\n')
        fo.write(f'{outBounder}\n')
        fo.write(f'{out1}\n')
        fo.write(f'{out2}\n')
        fo.write(f'{outBounder}\n')
        fo.write(f'{outBounder}\n\n')
    else:
        outBounder = "*" * len(out) * 2
        print(f'\n\n{outBounder}')
        print(f'{outBounder}')
        print(f'{out}')
        print(f'{outBounder}')
        print(f'{outBounder}\n')
        fo.write(f'\n\n{outBounder}\n')
        fo.write(f'{outBounder}\n')
        fo.write(f'{out}\n')
        fo.write(f'{outBounder}\n')
        fo.write(f'{outBounder}\n\n')


def RemoveStopwords(passage):
    new = ""
    for i in range(len(passage)):
        char = passage[i]
        if char not in punc:
            new += char

    return new

def Basic(passage, fo):
    StandardOutput('Basic', fo)

    global tp_title, fp_title, fn_title, tn_title
    global tp_dept, fp_dept, fn_dept, tn_dept
    global tp_time, fp_time, fn_time, tn_time
    global tp_count, fp_count, fn_count, tn_count
    global tp_school, fp_school, fn_school, tn_school
    global tp_kwds, fp_kwds, fn_kwds, tn_kwds
    global tp_nwds, fp_nwds, fn_nwds, tn_nwds

    pattern_title = re.compile(r'[\s]*[\S]+[\s]*\n')
    pattern_department = re.compile(r'发布部门：[\s]*[\S]+')
    pattern_time = re.compile(r'发布时间：[\s]*[\S]+')
    pattern_count = re.compile(r'浏览[\s]*[\d]+[\s]*次')
    pattern_school = re.compile(r'[\S]+学院')
    pattern_schoolforcalc = re.compile(r'[.]*学院[.]*\n')

    title = pattern_title.match(passage)
    dept = pattern_department.search(passage)
    time = pattern_time.search(passage)
    count = pattern_count.search(passage)
    school = pattern_school.search(passage)
    schoolforcalc = pattern_schoolforcalc.match(passage)

    kwds = ht.extract_keywords(passage, args.k, method="jieba_tfidf")
    newwords = ht.word_discover(passage, sort_by='score').index.tolist()

    set_kwds = set(kwds)
    set_nwds = set(newwords)
    set_new = set_nwds.difference(set_kwds)
    newwords = list(set_new)

    if title is not None:
        print(f'{"标题:     "} {title.group(0)[:-1]}')
        fo.write(f'{"标题:     "} {title.group(0)[:-1]}\n')
        tp_title += 1
    else:
        print(f'{"标题:     "} {"无"}')
        fo.write(f'{"标题:     "} {"无"}\n')
        tn_title += 1
    if dept is not None:
        print(f'{"发布部门: "} {dept.group(0)[5:]}')
        fo.write(f'{"发布部门: "} {dept.group(0)[5:]}\n')
        tp_dept += 1
    else:
        print(f'{"发布部门: "} {"无"}')
        fo.write(f'{"发布部门: "} {"无"}\n')
        tn_dept += 1
    if time is not None:
        print(f'{"发布时间: "} {time.group(0)[6:]}')
        fo.write(f'{"发布时间: "} {time.group(0)[6:]}\n')
        tp_time += 1
    else:
        print(f'{"发布时间: "} {"无"}')
        fo.write(f'{"发布时间: "} {"无"}\n')
        tn_time += 1
    if count is not None:
        print(f'{"浏览量:   "} {count.group(0)[3:-2]}')
        fo.write(f'{"浏览量:   "} {count.group(0)[3:-2]}\n')
        tp_count += 1
    else:
        print(f'{"浏览量:   "} {"无"}')
        fo.write(f'{"浏览量:   "} {"无"}\n')
        tn_count += 1
    if school is not None:
        print(f'{"学院:     "} {school.group(0)}')
        fo.write(f'{"学院:     "} {school.group(0)}\n')
        tp_count += 1
    else:
        print(f'{"学院:     "} {"无"}')
        fo.write(f'{"学院:     "} {"无"}\n')
        tn_count += 1
    if len(kwds) > 0:
        print(f'{"关键字:   "} {kwds}')
        fo.write(f'{"关键字:   "} {kwds}\n')
        tp_kwds += 1
    else:
        print(f'{"关键字:   "} {"无"}')
        fo.write(f'{"关键字:   "} {"无"}\n')
        tn_kwds += 1
    print(f'{"篇幅:     "} {len(passage)}')
    fo.write(f'{"篇幅:     "} {len(passage)}\n')
    if len(newwords) > 0:
        print(f'{"可能相关:  "} {newwords}')
        fo.write(f'{"可能相关:  "} {newwords}\n')
        tp_nwds += 1
    else:
        print(f'{"可能相关:  "} {"无"}')
        fo.write(f'{"可能相关:  "} {"无"}\n')
        tn_nwds += 1

def Entity(passage, fo):
    StandardOutput('Entity', fo)

    entities = ht.named_entity_recognition(passage)
    i = 0

    for key in entities:
        print(f'实体{i:3}:    {key}:{entities[key]}')
        fo.write(f'实体{i:3}:    {key}:{entities[key]}\n')
        i += 1


def EventDistill(passage, fo):
    StandardOutput('Event', fo)

    tripleRepresentation = ht.triple_extraction(passage)

    for i in range(len(tripleRepresentation)):
        print(f'{tripleRepresentation[i]}')
        fo.write(f'{tripleRepresentation[i]}\n')


def SentenceAnalyze(passage, fo):
    StandardOutput('Sentence', fo)

    for arc in ht.dependency_parse(passage):
        print(arc)
        fo.write(f'{arc}\n')


def SentsAnalyze(passage, fo):
    StandardOutput('Sents', fo)

    sentiment = senti.sentiment_calculate(passage)
    emo = emotion.emotion_count(passage)

    print(f'正向情感值: {sentiment["pos"]}')
    print(f'负向情感值: {sentiment["neg"]}')

    print(f'\n情绪分析结果:')
    print(f'好: {emo["好"]}')
    print(f'乐: {emo["乐"]}')
    print(f'哀: {emo["哀"]}')
    print(f'怒: {emo["怒"]}')
    print(f'惧: {emo["惧"]}')
    print(f'恶: {emo["恶"]}')
    print(f'惊: {emo["惊"]}')

    fo.write(f'正向情感值: {sentiment["pos"]}\n')
    fo.write(f'负向情感值: {sentiment["neg"]}\n')

    fo.write(f'\n情绪分析结果:\n')
    fo.write(f'好: {emo["好"]}\n')
    fo.write(f'乐: {emo["乐"]}\n')
    fo.write(f'哀: {emo["哀"]}\n')
    fo.write(f'怒: {emo["怒"]}\n')
    fo.write(f'惧: {emo["惧"]}\n')
    fo.write(f'恶: {emo["恶"]}\n')
    fo.write(f'惊: {emo["惊"]}\n')

def CutParagraph(passage, fo):
    StandardOutput('Cut', fo)

    predicted_paras = ht.cut_paragraphs(passage, num_paras=args.p)

    print(f'共 {len(predicted_paras)} 段, 分段结果: ')
    fo.write(f'共 {len(predicted_paras)} 段, 分段结果: \n')
    for i in range(len(predicted_paras)):
        print(f'第{i + 1}段: {predicted_paras[i]}')
        fo.write(f'第{i + 1}段: {predicted_paras[i]}\n')


def Question(passage, fo):
    StandardOutput('Question', fo)

    tripleRepresentation = ht.triple_extraction(passage)
    entities = ht.named_entity_recognition(passage)
    QA = NaiveKGQA(SVOs=tripleRepresentation, entity_type_dict=entities)

    if len(args.q) != 0:
        print(f'问题: {args.q}')
        print(f'回答: {QA.answer(args.q)}')
        fo.write(f'问题: {args.q}\n')
        fo.write(f'回答: {QA.answer(args.q)}\n')
    else:
        print(f'未进行提问')
        fo.write(f'未进行提问\n')


def Network(passage, filename, fo):
    StandardOutput('Network', fo)

    passage = ht.clean_text(passage)
    kwds = ht.extract_keywords(passage, args.k, method="jieba_tfidf")
    docs = ht.cut_sentences(passage)
    G = ht.build_word_ego_graph(docs, kwds[0], min_freq=3, other_min_freq=2, stopwords=punc)

    filter_G = nx.Graph()
    filter_G.clear()
    filter_G_list = []
    filter_G_list.clear()

    for edge in G.edges.items():
        if len(edge[0][0]) != 1 and len(edge[0][1]) != 1:
            filter_G_list.append((edge[0][0], edge[0][1], edge[1]["weight"]))
    filter_G.add_weighted_edges_from(filter_G_list)

    print(f'关系网络字典表示:')
    print(dict(filter_G.edges.items()))
    fo.write(f'关系网络字典表示:\n')
    fo.write(f'{dict(filter_G.edges.items())}\n')

    nowdir = os.path.join(filedir, '提取结果', '关系网络', str(filename) + '.jpg')
    if os.path.exists(nowdir):
        os.remove(nowdir)

    nx.draw_networkx(filter_G)
    plt.savefig(f'./提取结果/关系网络/{filename}.jpg', dpi=400)
    plt.clf()

def main(passages):
    print("Current Path in TextExtract ",os.getcwd())
    if args.specificdoc != "":
        original_passage = passage = read(args.specificdoc)

        fo = open(f'./提取结果/信息/{args.specificdoc}.txt', "w", encoding='utf-8')
        print(f'*****************************************')
        print(f'*****************************************')
        print(f'*****************************************')
        print(f'*****************************************')
        print(f'*****************************************')
        print(f'第 {args.specificdoc} 篇')
        print(f'*****************************************')
        print(f'*****************************************')
        print(f'*****************************************')
        print(f'*****************************************')
        print(f'*****************************************')

        if args.basic or args.all:
            Basic(passage, fo)

        passage = ht.clean_text(passage)
        passage = RemoveStopwords(passage)

        if args.ent or args.all:
            Entity(passage, fo)

        if args.sentence or args.all:
            SentenceAnalyze(passage, fo)

        if args.sents or args.all:
            SentsAnalyze(original_passage, fo)

        if args.event or args.all:
            EventDistill(passage, fo)

        if args.para or args.all:
            CutParagraph(original_passage, fo)

        if args.qa or args.all:
            Question(passage, fo)

        if args.network or args.all:
            Network(original_passage, args.specificdoc, fo)

        fo.close()
    
    for i in range(passages):
        if args.specificdoc != "":
            break
            

        else:
            original_passage = passage = read(i + 1)

            fo = open(f'./提取结果/信息/{i + 1}.txt', "w", encoding='utf-8')
            print(f'*****************************************')
            print(f'*****************************************')
            print(f'*****************************************')
            print(f'*****************************************')
            print(f'*****************************************')
            print(f'第 {i + 1} 篇')
            print(f'*****************************************')
            print(f'*****************************************')
            print(f'*****************************************')
            print(f'*****************************************')
            print(f'*****************************************')

            if args.basic or args.all:
                Basic(passage, fo)

            passage = ht.clean_text(passage)
            passage = RemoveStopwords(passage)

            if args.ent or args.all:
                Entity(passage, fo)

            if args.sentence or args.all:
                SentenceAnalyze(passage, fo)

            if args.sents or args.all:
                SentsAnalyze(original_passage, fo)

            if args.event or args.all:
                EventDistill(passage, fo)

            if args.para or args.all:
                CutParagraph(original_passage, fo)

            if args.qa or args.all:
                Question(passage, fo)

            if args.network or args.all:
                Network(original_passage, i + 1, fo)

            fo.close()

    fo = open(f'./提取结果/信息/结果指标.txt', "w", encoding='utf-8')
    CalcResult(fo)
    fo.close()

if __name__ == '__main__':
    
    if args.doc <= 100:
        main(args.doc)
    else:
        main(args.doc)
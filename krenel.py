# --coding:utf-8--
import os
import re
import numpy as np
import pandas as pd

from pip._vendor import chardet
import re

def process_sequence_file(file_content):
    # 正则表达式验证内容是否符合蛋白质序列、lncRNA序列、蛋白质GO、lncRNA表达谱的相关内容
    def is_protein_sequence(seq):
        return re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', seq, re.I) is not None

    def is_lncrna_sequence(seq):
        return re.match(r'^[AUCG]+$', seq, re.I) is not None

    def is_protein_go(content):
        return bool(re.search(r'GO:\d+', content))

    def is_lncrna_expression(line):
        """
        判断行是否为lncRNA表达谱数据
        :param line: 行字符串
        :return: 如果行符合lncRNA表达谱格式返回True，否则返回False
        """
        pattern = re.compile(
            r'^\S+\t\d+\t\d+\t\S+\t\d+\t[+-]\t\d+\t\d+\t\d+\t[\d,]+\t\d+\t[\d,]+$'
        )
        return bool(pattern.match(line))

    lines = file_content.splitlines()
    sequences, metadata = [], []
    for line in lines:
        if is_protein_sequence(line) or is_lncrna_sequence(line):
            sequences.append(line)
        elif is_protein_go(line) or is_lncrna_expression(line):
            metadata.append(line)
        else:
            metadata.append(line)

    return sequences, metadata


# 读取CSV、XLSX、XLS和TSV格式的序列
def read_tabular_file(file, sequences_type = None):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.tsv'):
        df = pd.read_csv(file, sep='\t')
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file)
    else:
        return [], [], None
    # 检查内容是否为lncRNA表达谱数据或蛋白质GO数据
    if is_lncRNA_expression(df):
        sequences_type = "lncRNA_exp"
        sequences, metadata = process_lncRNA_expression(df)
    elif is_protein_go(df):
        sequences_type = "protein_go"
        sequences, metadata = process_protein_go(df)
    else:
        # 处理常规序列数据
        sequences, metadata = process_general_sequence_data(df)
    return sequences, metadata, sequences_type

def is_lncRNA_expression(df):
    """
    判断一个DataFrame是否为lncRNA表达谱数据。
    """
    pattern = re.compile(
        r'^\S+\t\d+\t\d+\t\S+\t\d+\t[+-]\t\d+\t\d+\t\d+\t[\d,]+\t\d+\t[\d,]+$'
    )

    def match_row(row):
        row_str = '\t'.join(map(str, row))
        return bool(pattern.match(row_str))

    return any(df.apply(match_row, axis=1))

def process_lncRNA_expression(df):
    sequences = df.iloc[:, 1:].values.tolist()
    metadata = df.iloc[:, 0].values.tolist()
    return sequences, metadata

def is_protein_go(df):
    """
    判断一个DataFrame是否为蛋白质GO数据。
    """
    # 蛋白质GO数据通常具有特定的列名称或包含GO术语
    # 这里简单地假设列名或数据中包含 "GO" 关键词
    return any("GO" in str(col).upper() for col in df.columns) or df.apply(
        lambda x: any(isinstance(i, str) and "GO:" in i for i in x), axis=1
    ).any()

def process_protein_go(df):
    sequences = []
    metadata = []

    for index, row in df.iterrows():
        row_data = row.tolist()
        go_entries = [entry for entry in row_data if isinstance(entry, str) and "GO:" in entry]
        other_data = [entry for entry in row_data if not (isinstance(entry, str) and "GO:" in entry)]

        if go_entries:
            sequences.append(go_entries)
            metadata.append(other_data)

    return sequences, metadata


def process_general_sequence_data(df):
    """
    处理常规的序列数据。
    """
    sequences, metadata = [], []
    # 打印 DataFrame 的信息
    print("DataFrame Info:")
    print(df.info())
    print("DataFrame Head:")
    print(df.head())

    # 检查是否有列名或行名为 "Sequence"
    if "Sequence" in df.columns:
        sequences = df["Sequence"].dropna().tolist()
        metadata = df.drop(columns=["Sequence"]).apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1).tolist()
        metadata = [meta for meta in metadata if meta.strip()]  # 过滤掉仅包含空格或空字符串的元数据
    elif "Sequence" in df.index:
        sequences = df.loc["Sequence"].dropna().tolist()
        metadata = df.drop(index=["Sequence"]).apply(lambda x: ', '.join(x.dropna().astype(str)), axis=0).tolist()
        metadata = [meta for meta in metadata if meta.strip()]  # 过滤掉仅包含空格或空字符串的元数据
    # 检查是否有列是序列
    if any(is_sequence_column(df[col]) for col in df.columns):
        sequence_columns = [col for col in df.columns if is_sequence_column(df[col])]
        sequences = df[sequence_columns[0]].dropna().tolist()
        metadata = df.drop(columns=sequence_columns).apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1).tolist()
        metadata = [meta for meta in metadata if meta.strip()]  # 过滤掉仅包含空格或空字符串的元数据
        return sequences, metadata
    else:
    # 检查是否有行是序列
        if any(df.apply(is_sequence_column, axis=1)):
            sequence_rows = df.apply(is_sequence_column, axis=1)
            sequences = df[sequence_rows].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1).tolist()
            metadata = df[~sequence_rows].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1).tolist()
            metadata = [meta for meta in metadata if meta.strip()]
    print("tttttt")
    print(sequences)
    print(metadata)

    return sequences, metadata


def is_sequence_column(column):
    """
    判断一个列是否为序列列。
    :param column: DataFrame 列
    :return: 如果该列可能为序列列返回 True，否则返回 False
    """
    if column.dtype == 'object' and all(column.str.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$|^[AUCG]+$', na=False)):
        return True
    return False

def read_text_file(file):
    """
    读取文本文件并返回序列列表。
    :param file: 上传的文件对象
    :return: 序列列表
    """
    encoding = detect_encoding(file)
    file_content = file.read().decode(encoding)
    sequences = file_content.splitlines()
    sequences = [seq.strip() for seq in sequences if seq.strip()]
    return sequences

# 动态生成核矩阵
def generate_kernel_matrix(sequences, kernel_type, sequences_type):
    n = len(sequences)
    kernel_matrix = np.zeros((n, n))

    if kernel_type == "swl" or kernel_type == "swp":
        for i in range(n):
            for j in range(i, n):
                score = smith_waterman(sequences[i], sequences[j])
                kernel_matrix[i][j] = kernel_matrix[j][i] = score
    elif kernel_type == "ct" or kernel_type == "ps":
        for i in range(n):
            for j in range(i, n):
                vec_i, vec_j = feature_vector(sequences[i]), feature_vector(sequences[j])
                score = np.exp(-1.0 * np.sum((vec_i - vec_j) ** 2))
                kernel_matrix[i][j] = kernel_matrix[j][i] = score
    elif kernel_type == "ep" and sequences_type == "lncRNA_exp":
        expressions = np.random.rand(n, 24)
        for i in range(n):
            for j in range(i, n):
                score = np.exp(-1.0 * np.sum((expressions[i] - expressions[j]) ** 2))
                kernel_matrix[i][j] = kernel_matrix[j][i] = score
    elif kernel_type == "go" and sequences_type == "protein_go":
        go_terms = [set(seq) for seq in sequences]
        for i in range(n):
            for j in range(i, n):
                score = jaccard_similarity(go_terms[i], go_terms[j])
                kernel_matrix[i][j] = kernel_matrix[j][i] = score
    else:
        raise ValueError("Invalid kernel type for the given sequences type")
    return kernel_matrix


# Smith-Waterman简化实现
def smith_waterman(seq1, seq2):
    return len(set(seq1) & set(seq2))


# 简化的特征向量
def feature_vector(seq, max_length=100):
    """
    简化的特征向量生成函数，将序列转换为固定长度的数值特征向量。
    :param seq: 输入序列
    :param max_length: 特征向量的最大长度
    :return: 特征向量
    """
    vec = np.zeros(max_length)
    for i, char in enumerate(seq):
        if i >= max_length:
            break
        vec[i] = ord(char)
    return vec


# Jaccard相似性
def jaccard_similarity(set1, set2):
    return float(len(set1 & set2)) / len(set1 | set2)

def ensure_dir(directory):
    """
    确保目录存在。如果不存在，则创建它。
    :param directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
# 保存核矩阵到文件
def save_kernel_matrix(kernel_matrix, filename):
    ensure_dir(os.path.dirname(filename))
    df = pd.DataFrame(kernel_matrix)
    df.to_csv(filename, index=False)

def save_metadata(metadata, filename):
    ensure_dir(os.path.dirname(filename))
    with open(filename, 'w') as file:
        for item in metadata:
            file.write(f"{item}\n")

def file_exists(filename):
    """
    检查文件是否存在。
    :param filename: 文件名
    :return: 如果文件存在返回True，否则返回False
    """
    return os.path.exists(filename)

def detect_encoding(file):
    """
    检测文件的编码。
    :param file: 上传的文件对象
    :return: 文件编码
    """
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    file.seek(0)  # 重新设置文件指针到文件开头
    return encoding

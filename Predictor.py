import numpy as np
import pandas as pd
import streamlit as st
from Bio import SeqIO
import math
import csv
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler
import os

st.title(
    """
               ************** m5C-iEnsem  **************

"""
)
image = Image.open('Flow_Chart.jpg')
st.image(image)

def seqToMat(seq):
    encoder = ['a', 'c', 'u', 'g']
    lent = len(seq)
    n = int(math.ceil(math.sqrt(lent)))
    seqMat = [[0 for x in range(n)] for y in range(n)]
    seqiter = 0
    for i in range(n):
        for j in range(n):
            if seqiter < lent:
                try:
                    aa = int(encoder.index(seq[seqiter]))
                except ValueError:
                    exit(0)
                else:
                    seqMat[i][j] = aa
                seqiter += 1
    return seqMat

def frequencyVec4(seq):
    encoder = ['a', 'c', 'u', 'g']
    fv = [0 for _ in range(4)]
    for i in range(4):
        fv[i] = seq.count(encoder[i])
    return fv

def rawMoments(mat, order):
    n = len(mat)
    rawM = []
    sum_val = 0
    for i in range(order + 1):
        for j in range(order + 1):
            if i + j <= order:
                for p in range(n):
                    for q in range(n):
                        sum_val += (((p + 1) ** i) * ((q + 1) ** j) * int(mat[p][q]))
                rawM.append(sum_val)
                sum_val = 0
    return rawM

def centralMoments(mat, order, xbar, ybar):
    n = len(mat)
    centM = []
    sum_val = 0
    for i in range(order + 1):
        for j in range(order + 1):
            if i + j <= order:
                for p in range(n):
                    for q in range(n):
                        sum_val += ((((p + 1) - xbar) ** i) * (((q + 1) - ybar) ** j) * mat[p][q])
                centM.append(sum_val)
                sum_val = 0
    return centM

def hahnMoments(mat, order):
    N = len(mat)
    hahnM = []
    for i in range(order + 1):
        for j in range(order + 1):
            if i + j <= order:
                answer = hahnMoment(i, j, N, mat)
                hahnM.append(answer)
    return hahnM

def hahnMoment(m, n, N, mat):
    value = 0.0
    for x in range(N):
        for y in range(N):
            value += mat[x][y] * hahnProcessor(x, m, N) * hahnProcessor(y, n, N)
    return value

def hahnProcessor(x, n, N):
    return hahnPol(x, n, N) * math.sqrt(roho(x, n, N))

def hahnPol(x, n, N):
    answer = 0.0
    ans1 = pochHammer(N - 1.0, n) * pochHammer(N - 1.0, n)
    ans2 = 0.0
    for k in range(n + 1):
        ans2 += math.pow(-1.0, k) * ((pochHammer(-n, k) * pochHammer(-x, k) * pochHammer(2 * N - n - 1.0, k)))
    answer = ans1 + ans2
    return answer

def roho(x, n, N):
    return math.exp(math.lgamma(n + 1.0)) * math.exp(math.lgamma(n + 1.0)) * pochHammer((n + 1.0), N)

def pochHammer(a, k):
    answer = 1.0
    for i in range(k):
        answer *= (a + i)
    return answer

def calcFV(seq):
    encoding1 = [cal_dibase_index(seq[i:i+2]) for i in range(0, len(seq)-1, 2)]
    encoding2 = [cal_tribase_index(seq[i:i+3]) for i in range(0, len(seq)-2, 3)]
    fv = [0 for _ in range(522)]
    fvIter = 0
    myMat = seqToMat(seq)
    myRawMoments = rawMoments(myMat, 3)
    xbar = myRawMoments[4]
    ybar = myRawMoments[1]
    myCentralMoments = centralMoments(myMat, 3, xbar, ybar)
    myHahnMoments = hahnMoments(myMat, 3)
    myFrequencyVec4 = frequencyVec4(seq)

    for ele in myRawMoments + myCentralMoments + myHahnMoments + myFrequencyVec4:
        fv[fvIter] = ele
        fvIter += 1

    return fv

def input_seq():
    st.subheader("Input Sequence of any length")
    sequence1 = st.text_area("Sequence Input", value="CGCCUCCCACGCGGGAGACCCGGGUUCAAUUCCCGGCCAAU", height=200)

    if st.button("Submit"):
        count = [i for i, char in enumerate(sequence1) if char == 'C']
        keeper = [sequence1[max(0, c-20):min(len(sequence1), c+21)] for c in count]

        allFVs = [calcFV(seq.lower()) for seq in keeper]

        with open('IISequence_FVs_for_test.csv', mode='w') as fvFile:
            fvWriter = csv.writer(fvFile)
            fvWriter.writerows(allFVs)

        # CSV file handling
        csv_file = "IISequence_FVs_for_test.csv"
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            df = pd.read_csv(csv_file, sep=",", header=None)
            W = df.iloc[:, :].values
        else:
            st.error("CSV file is missing or empty!")
            return

        std_scale = StandardScaler().fit(W)
        W = std_scale.transform(W)
        load_model = pickle.load(open('Final_model.pkl', 'rb'))
        pred = load_model.predict(W)
        output_proba = load_model.predict_proba(W)[:, 1]

        for i, proba in enumerate(output_proba):
            st.subheader(f"Site Number = {count[i]}")
            st.write(keeper[i])
            if proba > 0.7:
                st.info("Output = 5-Methylcytosine Site")
            else:
                st.info("Output = Non-5-Methylcytosine Site")

input_seq()

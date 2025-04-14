import pandas as pd
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
import numpy as np

def calc_CDR(df):
    total_num_exams = len(df)
    cancer_detected_exams = ((df["exam_level_PIRADS_"].isin(["3", "4", "5"])) & (df["csPCa"]==1)).sum()
    CDR = cancer_detected_exams/total_num_exams
    CDR_CIs = proportion_confint(cancer_detected_exams, total_num_exams)
    return CDR, CDR_CIs, cancer_detected_exams, total_num_exams

def calc_AIR(df):
    total_num_exams = len(df)
    abnormal_exams = (df["exam_level_PIRADS_"].isin(["3", "4", "5"])).sum()
    AIR = abnormal_exams/total_num_exams
    AIR_CIs = proportion_confint(abnormal_exams, total_num_exams)
    return AIR, AIR_CIs, abnormal_exams, total_num_exams

def calc_PPV(df):
    cancer_detected_exams = ((df["exam_level_PIRADS_"].isin(["3", "4", "5"])) & (df["csPCa"]==1)).sum()
    abnormal_exams_with_path = (df["exam_level_PIRADS_"].isin(["3", "4", "5"]) & (df["Post_MRI_Dx_from_path_"]!="No pathology")).sum()
    PPV = cancer_detected_exams/abnormal_exams_with_path
    PPV_CIs = proportion_confint(cancer_detected_exams, abnormal_exams_with_path)
    return PPV, PPV_CIs, cancer_detected_exams, abnormal_exams_with_path

def calc_bx_rate(df):
    abnormal_exams_with_path = ((df["exam_level_PIRADS_"].isin(["3", "4", "5"])) & (df["Post_MRI_Dx_from_path_"]!="No pathology")).sum()
    abnormal_exams = (df["exam_level_PIRADS_"].isin(["3", "4", "5"])).sum()
    bx_prop = abnormal_exams_with_path/abnormal_exams
    bx_prop_CIs = proportion_confint(abnormal_exams_with_path, abnormal_exams)
    return bx_prop, bx_prop_CIs, abnormal_exams_with_path, abnormal_exams
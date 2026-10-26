# Patient-Admission-and-Discharge-Trend-Analysis
1. Introduction & Problem Statement
1.1 Background
Hospitals continuously generate records of who is admitted, through which channel, how long they stay, and where they go afterwards. Hospital administrators need to turn this raw record trail into planning information: how many beds will be needed, which patient groups occupy them longest, and which patients are most likely to return soon after discharge. Raw encounter tables of more than one hundred thousand rows cannot be interpreted by inspection, so Python-based data analytics is used to clean, summarise and visualise them.
1.2 Problem Statement
A hospital bed-management and capacity-planning team needs a clear, data-backed picture of patient flow: which admission types and sources bring in the most patients, how long different patient segments occupy a bed, where patients are discharged to, and which segments return within 30 days. This project analyses 101,766 inpatient encounters from 130 US hospitals to answer these questions and to give a concrete, quantified basis for bed-occupancy and staffing decisions.
Why it matters: length of stay (LOS) multiplied by admission volume is the direct driver of bed occupancy; discharge destination determines downstream care demand; and early readmissions both consume capacity and signal gaps in discharge planning. Expected benefit: administrators can identify the highest-volume entry channels, the longest-stay segments and the highest-risk discharge groups. Real-world application: bed-occupancy planning, staff allocation, discharge-planning priorities and readmission-reduction programmes.
1.3 Objectives
1.	To analyse patient admission patterns by admission type, admission source, age group and gender, and identify the dominant (peak) admission categories.
2.	To analyse discharge patterns and the flow from admission type to discharge destination.
3.	To quantify the average length of hospital stay (with 95% confidence intervals) for at least five patient segments.
4.	To identify patient segments with a higher workload burden (long stays) and a higher 30-day readmission rate.
5.	To create clear, meaningful Python visualisations of admission, discharge and readmission patterns, supported by basic statistical tests.
1.4 Scope Note: Adjustment from the Synopsis
The approved synopsis proposed daily, monthly and yearly admission/discharge trends and a "top 3 peak admission months" outcome. During dataset inspection (Section 2) it was confirmed that the Diabetes 130-US Hospitals dataset is de-identified and contains no admission or discharge date/time columns – the only time-related field is time_in_hospital (length of stay in days). Calendar-based trend analysis is therefore not possible with this dataset. Following the spirit of the synopsis, "peak" periods are interpreted as the peak categories of admission and discharge (highest-volume admission type, source, age band and discharge destination), and "trend" analysis is performed across patient segments rather than across calendar time. The measurable outcome is correspondingly restated in Section 7.2: exact counts for the top three admission categories and mean LOS reported to two decimals for more than five segments. Calendar-based analysis is listed as future scope using a date-stamped dataset.
1.5 Tools and Technologies
Python 3, NumPy, Pandas (data handling), Matplotlib and Seaborn (visualisation), SciPy (statistical tests); VS Code / Jupyter environment. All libraries are among those mandated for the course.
2. Dataset Description & Source
Table 1: Dataset summary
Item	Details
Name	Diabetes 130-US Hospitals for Years 1999–2008
Source	UCI Machine Learning Repository (Center for Clinical and Translational Research, Virginia Commonwealth University); accessed via Kaggle
Type / Format	Public, open dataset; CSV file (diabetic_data.csv)
Raw size	101,766 patient encounters × 50 attributes (verified by loading the file)
Unique patients	71,518 (16,773 patients have more than one encounter)
Coverage	130 US hospitals and integrated delivery networks, 10 years (1999–2008); inpatient stays of 1–14 days where diabetes was recorded as a diagnosis
Quality criteria	≥ 500 records ✔, ≥ 5 attributes ✔, missing values concentrated in a few non-essential columns (justified in Section 3)

This resolves the synopsis remark requesting the exact record count and feature list: the file contains exactly 101,766 rows and 50 columns, listed below by group.
Table 2: Attribute groups
Group	Attributes	Type
Identifiers	encounter_id, patient_nbr	Numeric ID
Demographics	race, gender, age (10-year bands), weight	Categorical
Admission / discharge	admission_type_id, admission_source_id, discharge_disposition_id, payer_code, medical_specialty	Coded categorical
Utilisation & workload	time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses	Numeric (count/days)
Diagnoses	diag_1, diag_2, diag_3 (ICD-9 codes)	Categorical code
Diabetes testing & drugs	max_glu_serum, A1Cresult, 23 medication columns, change, diabetesMed	Categorical
Outcome	readmitted (NO / >30 / <30 days)	Categorical target

Limitation – no dates. As noted in Section 1.4, the dataset has no calendar timestamps, so time_in_hospital (days) is the only temporal measure. Ethics: the data are publicly available and fully de-identified; no attempt was made to re-identify patients, and all sources are cited in Section 9.
3. Data Cleaning & Preprocessing
The raw file was loaded with the "?" placeholder interpreted as a missing value. Checks showed no fully duplicated rows and no repeated encounter_id values. The missing-value profile is shown in Figure 1 and the cleaning steps are summarised in Table 3.
 
Figure 1: Percentage of missing values per column (raw data)
Table 3: Cleaning and preprocessing steps
#	Step	Reason	Result
1	Replace "?" with NaN	Dataset encodes missing as "?"	Proper missing-value detection
2	Drop weight	96.86% missing – unusable	Column removed
3	Drop payer_code	39.56% missing; not needed for objectives	Column removed
4	Drop examide, citoglipton	Single constant value in all rows	2 columns removed
5	Fill medical_specialty (49.08%) and race (2.23%) with "Unknown"	Keeps all records; missingness is itself informative	No rows lost
6	Fill max_glu_serum (94.75%) and A1Cresult (83.28%) with "Not tested"	Missing means the test was not performed (per data documentation)	Meaningful category
7	Remove 3 rows with gender "Unknown/Invalid"	Invalid category	101,766 → 101,763 rows
8	Map admission_type_id, admission_source_id, discharge_disposition_id to readable groups	Numeric codes are not interpretable; many codes are duplicates of "NULL / Not available"	admission_type, admission_source, discharge_group
9	Group diag_1 ICD-9 codes into 9 disease families	>700 raw codes are too sparse	primary_diag_group
10	Create readmit_30 (1 if "<30") and age_mid (band midpoint)	Binary outcome and numeric age for correlation	2 derived columns
11	Exclude 2,423 "expired/hospice" discharges from readmission analysis only	Patients who died or entered hospice cannot be readmitted; including them would understate rates	99,340 encounters for readmission analysis

Result: the cleaned dataset has 101,763 rows and 52 columns and was saved as cleaned_diabetic_data.csv. No numeric outliers were removed: 2,252 encounters (2.2%) exceed the IQR upper fence of 12 days, but these are genuine long stays (maximum 14 days, the dataset's cap) and are relevant to bed-occupancy planning. A note on independence: 16,773 patients appear more than once, so encounters are not strictly independent; this is treated as a limitation (Section 8).
raw = pd.read_csv('diabetic_data.csv', na_values=['?'])
df  = raw.drop(columns=['weight','payer_code','examide','citoglipton'])
df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
df[['max_glu_serum','A1Cresult']] = df[['max_glu_serum','A1Cresult']].fillna('Not tested')
df = df[df.gender != 'Unknown/Invalid']
df['readmit_30'] = (df.readmitted == '<30').astype(int)
4. Exploratory Data Analysis (EDA)
4.1 Length of stay (workload measure)
Across 101,763 encounters the mean length of stay is 4.40 days (95% CI 4.38–4.41), median 4 days, mode 3 days, standard deviation 2.98 and range 1–14 days. The distribution is right-skewed (skewness = 1.134), so the median is a more representative "typical" stay than the mean, while the long tail is what stresses bed capacity.
4.2 Admission patterns
Table 4: Admissions by admission type (peak category first)
Admission type	Encounters	% of total	Mean LOS (days)
Emergency	53,988	53.05%	4.38
Elective	18,868	18.54%	4.32
Urgent	18,480	18.16%	4.61
Not available	10,396	10.22%	4.24
Trauma center	21	0.02%	4.86
Newborn	10	0.01%	3.20

Table 5: Admissions by admission source
Admission source	Encounters	% of total	Mean LOS (days)
Emergency room	57,492	56.50%	4.39
Physician referral	29,564	29.05%	4.26
Other / unknown	7,096	6.97%	4.16
Transfer: hospital	3,187	3.13%	5.82
Transfer: other facility	2,264	2.22%	4.43
Clinic referral	1,104	1.08%	4.68
Transfer: SNF	855	0.84%	5.83
HMO referral	187	0.18%	4.82

Emergency admissions dominate: 53,988 encounters (53.05%), followed by Elective (18,868; 18.54%) and Urgent (18,480; 18.16%). The emergency room is the entry point for 56.50% of admissions and physician referral for 29.05%. About 10.22% of admission types are not recorded.
4.3 Demographics
Females account for 54,708 encounters (53.76%) and males 47,055 (46.24%). The population is elderly: patients aged 60 and above make up 68,538 encounters (67.35%), and the single largest band is 70–80 years (26,066; 25.61%). Children under 20 contribute only 0.84%.
4.4 Discharge patterns
Table 6: Discharge destination (grouped)
Discharge group	Encounters	% of total	Mean LOS (days)
Home	60,232	59.19%	3.70
Facility transfer / care	20,765	20.41%	5.57
Home with health service	13,010	12.78%	5.45
Other / unknown	4,710	4.63%	5.04
Expired / hospice	2,423	2.38%	5.08
Left against advice	623	0.61%	2.84

Home is the most common destination (59.19%), but 33.19% of patients need continuing care (facility transfer or home with health services), which creates demand beyond the hospital bed. 2,423 encounters (2.38%) ended in death or hospice.
4.5 Diagnoses and prior utilisation
Circulatory diseases are the leading primary diagnosis (30,436; 29.91%), followed by respiratory (14.17%), digestive (9.31%) and diabetes-coded primary diagnoses (8.61%). Most patients had no prior inpatient visit in the preceding year (66,242 of 99,340 encounters in the readmission set).
4.6 Readmission outcome
Among the 99,340 encounters eligible for readmission analysis, 52.87% were not readmitted, 35.74% were readmitted after more than 30 days and 11.39% (11,314 encounters) were readmitted within 30 days.
5. Data Visualization & Observations
 
Figure 2: Distribution of length of stay (dashed line = mean)
Observation: most stays are short (1–4 days); stays of 2 and 3 days are the most frequent, and the frequency declines steadily towards the 14-day maximum, giving a long right tail.
 
Figure 3: Admissions by age group and gender
Observation: admissions rise steeply with age and peak at 70–80 years; the number of patients under 40 is very small. Female admissions are slightly higher than male admissions.
 
Figure 4: Admissions by admission type and by admission source
Observation: emergency admissions and emergency-room entry are the clearly dominant "peak" categories; elective and urgent admissions are similar in volume to each other. Hospital-to-hospital transfers are the third largest identified source.
 
Figure 5: Grouped discharge destinations
Observation: the majority of patients go home, but facility transfers and home-health discharges together form roughly one third of all discharges.
 
Figure 6: Admission type versus discharge destination (row percentages)
Observation: patients admitted as Urgent or Elective are discharged home in 62–63% of cases, compared with 58.7% for Emergency admissions and only 49.0% where admission type is not recorded. Emergency patients are more likely to die or enter hospice (3.0% vs 1.1% for elective).
 
Figure 7: Length of stay by age group and by admission type
Observation: the median LOS rises from about 2 days in children to 4 days in patients over 60, and the interquartile range widens, indicating more variable stays in older patients. Urgent admissions have a slightly higher mean stay than Emergency and Elective.
 
Figure 8: Mean length of stay with 95% CI by discharge destination
Observation: patients discharged to a facility (5.57 days) or home with health services (5.45 days) stay about 1.8–1.9 days longer than those discharged home (3.70 days). Those leaving against medical advice stay the shortest (2.84 days).
 
Figure 9: Primary diagnosis groups
Observation: circulatory conditions alone account for nearly 30% of admissions, making them the single biggest source of bed demand in this population. Neoplasm admissions have the longest mean stay (5.36 days) and musculoskeletal the shortest (3.92 days).
 
Figure 10: Readmission outcome
 
Figure 11: 30-day readmission rate by age, prior inpatient visits and discharge destination
Observation: the 30-day readmission rate increases steadily with the number of prior inpatient visits (8.6% with none to 31.4% with four or more) and is highest for patients discharged to a facility (16.4%) and lowest for those discharged home (9.3%). Rates are low in young children (1.9%) and ~12% for patients aged 70 and above. The 14.3% rate in the 20–30 band is based on a small group (1,649 encounters) and should be interpreted cautiously.
 
Figure 12: Correlation matrix of numeric attributes
Observation: length of stay correlates most strongly with the number of medications (r = 0.47), then lab procedures (0.32), number of diagnoses (0.22) and procedures (0.19). Prior outpatient and emergency visits show essentially no correlation with LOS.
6. Statistical Analysis (Basic)
Because LOS is right-skewed, non-parametric tests (Kruskal–Wallis, Mann–Whitney) were used alongside the parametric equivalents. With more than 100,000 records even tiny differences become statistically significant, so effect sizes (Cramér's V, correlation r) are reported to judge practical importance. Significance level α = 0.05.
6.1 Descriptive statistics and confidence intervals
Table 7: Mean LOS with 95% confidence interval for selected patient segments (n ≥ 500)
Segment	n	Mean	Median	95% CI (days)	± half-width
Admission type: Elective	18,868	4.32	3.0	4.28 – 4.36	0.044
Admission type: Emergency	53,988	4.38	4.0	4.35 – 4.40	0.025
Admission type: Not available	10,396	4.24	3.0	4.19 – 4.30	0.056
Admission type: Urgent	18,480	4.61	4.0	4.57 – 4.65	0.044
Discharge: Expired / hospice	2,423	5.08	4.0	4.94 – 5.22	0.141
Discharge: Facility transfer / care	20,765	5.57	5.0	5.53 – 5.62	0.043
Discharge: Home	60,232	3.70	3.0	3.68 – 3.72	0.021
Discharge: Home with health service	13,010	5.45	5.0	5.40 – 5.50	0.053
Discharge: Left against advice	623	2.84	2.0	2.65 – 3.03	0.187
Discharge: Other / unknown	4,710	5.04	4.0	4.95 – 5.13	0.093
Age: [40-50)	9,685	4.04	3.0	3.98 – 4.10	0.058
Age: [60-70)	22,482	4.38	4.0	4.34 – 4.42	0.039
Age: [70-80)	26,066	4.59	4.0	4.55 – 4.63	0.037
Age: [80-90)	17,197	4.81	4.0	4.76 – 4.85	0.045

6.2 Hypothesis tests
Table 8: Summary of hypothesis tests
Question	Test	Statistic	p-value	Conclusion
LOS differs by admission type (Emerg./Urgent/Elective)	One-way ANOVA	F = 53.03	p < 0.001	Significant
Same, non-parametric	Kruskal–Wallis	H = 132.9	p < 0.001	Significant
LOS differs by age group	Kruskal–Wallis	H = 1582.8	p < 0.001	Significant
LOS differs by discharge group	Kruskal–Wallis	H = 9641.8	p < 0.001	Significant
LOS: readmitted <30 d (4.77) vs others (4.33)	Welch t-test	t = 14.53	p < 0.001	Significant
Same, non-parametric	Mann–Whitney U	—	p < 0.001	Significant
Readmission <30 d vs age group	Chi-square (df = 9)	χ² = 133.6	p < 0.001	V = 0.037 (very small)
Readmission <30 d vs admission type	Chi-square (df = 5)	χ² = 30.0	p < 0.001	V = 0.017 (negligible)
Readmission <30 d vs discharge group	Chi-square (df = 4)	χ² = 808.0	p < 0.001	V = 0.090 (small)
Readmission <30 d vs primary diagnosis	Chi-square (df = 9)	χ² = 85.5	p < 0.001	V = 0.029 (very small)
Readmission <30 d vs gender	Chi-square (df = 1)	χ² = 0.63	p = 0.428	Not significant

6.3 Correlation analysis
Table 9: Correlation of length of stay with numeric attributes (Pearson r)
Attribute	r	p-value	Strength
Number of medications	0.466	< 0.001	Moderate
Number of lab procedures	0.318	< 0.001	Weak–moderate
Number of diagnoses	0.220	< 0.001	Weak
Age (band midpoint)	0.108	< 0.001	Weak
Prior inpatient visits (vs 30-day readmission)	0.168	< 0.001	Weak

The Spearman rank correlation between LOS and number of medications (ρ = 0.465) is almost identical to the Pearson value, confirming the relationship is robust to the skewed LOS distribution.
7. Key Insights & Interpretation
7.1 Insights
1.	Emergency care is the main driver of admissions. 53.05% of encounters are emergency admissions and 56.50% enter through the ER, so bed availability and ER-to-ward flow are the critical operational points.
2.	Elderly patients dominate demand. 67.35% of encounters are aged 60+, and older patients stay longer (mean 2.55 days for ages 0–10 vs 4.81 days for ages 80–90; Kruskal–Wallis p < 0.001).
3.	Discharge destination is closely tied to bed-days. Patients discharged to facilities or with home-health services stay about 1.8–1.9 days longer than those discharged home, indicating that complex cases drive occupancy and that delays arranging post-acute care may prolong stays.
4.	Workload is measurably linked to treatment intensity. The number of medications (r = 0.47) and lab procedures (r = 0.32) are the strongest correlates of LOS; these can act as early signals for expected long stays.
5.	Prior inpatient history is the strongest simple readmission signal. The 30-day readmission rate rises from 8.6% (no prior inpatient visit) to 31.4% (four or more), and discharge to a facility carries a 16.4% rate versus 9.3% for home.
6.	Gender has no meaningful effect on readmission (χ² p = 0.43), while age, discharge group and diagnosis have statistically significant but small effects (Cramér's V ≤ 0.09). Statistical significance should therefore not be over-read in a dataset of this size.
7.	Readmitted patients had longer index stays (mean 4.77 vs 4.33 days), suggesting that sicker patients both occupy beds longer and return sooner.
7.2 Achievement of the stated outcome
Table 10: Synopsis outcome versus delivered result
Synopsis outcome	Delivered (adapted to dataset)
Exact record count and feature list	101,766 records × 50 attributes; 101,763 × 52 after cleaning
Top 3 peak admission periods with exact counts	Calendar months not available. Top 3 admission categories with exact counts: Emergency 53,988; Elective 18,868; Urgent 18,480. Top 3 sources: ER 57,492; physician referral 29,564; other/unknown 7,096
Average LOS for ≥ 5 segments within 0.1 day precision	Mean LOS with 95% CI for 14 segments (Table 7). 95% CI half-width ≤ 0.1 day for all segments with n > ~3,000, e.g. Emergency (±0.025), Urgent (±0.044), Elective (±0.044), Home (±0.021), Facility transfer (±0.043), Home with health service (±0.053)
Basis for bed-occupancy and staffing decisions	Quantified admission volume, LOS, discharge-flow and readmission risk by segment (Sections 4–6)

7.3 Practical interpretation for hospital planning
•	Plan bed capacity and ER-to-ward capacity around emergency inflow, since it contributes more than half of admissions.
•	Allocate discharge-planning and case-management resources early for patients likely to need facility or home-health placement, as they occupy beds for roughly 1.8–1.9 extra days.
•	Use prior inpatient visits and discharge destination as simple flags for follow-up calls or transitional-care programmes to target the ~11% of patients who return within 30 days.
8. Conclusion & Future Scope
8.1 Conclusion
This project cleaned and analysed 101,766 real hospital encounters using Python. The analysis showed that hospital workload in this population is driven by emergency admissions (53%), entered mainly through the emergency room (56.5%), and by elderly patients (67% aged 60+) who stay about two days longer than young patients. Mean length of stay is 4.40 days (median 4). Patients who need post-acute care stay roughly 1.8–1.9 days longer than those discharged home, and about 11.39% of eligible patients return within 30 days, with the risk strongly increasing with prior inpatient visits and facility discharge. Statistical tests confirmed these differences, with effect sizes indicating that practical importance is largest for discharge destination, prior inpatient history and treatment intensity.
8.2 Limitations
•	The dataset has no admission or discharge dates, so calendar-time trends (daily/monthly/yearly) and seasonality could not be analysed.
•	The data are from 1999–2008 and cover diabetic patients in US hospitals; findings may not generalise to current practice or to Indian hospitals.
•	About 23% of patients have multiple encounters, so observations are not fully independent.
•	Several admission/discharge codes are "not available/unknown", and the age is available only in 10-year bands.
•	The analysis is descriptive and associational; it does not establish causation.
8.3 Future Scope
•	Repeat the analysis on a date-stamped admission/discharge dataset to obtain daily, monthly and seasonal trends and peak-month identification.
•	Build a predictive model (e.g. logistic regression or random forest using scikit-learn) for 30-day readmission and for length-of-stay prediction.
•	Develop an interactive Plotly or Streamlit dashboard for hospital administrators.
•	Use patient-level (grouped) validation to account for repeated encounters, and compare results with recent public hospital data.
9. References
[1]	B. Strack, J. P. DeShazo, C. Gennings, J. L. Olmo, S. Ventura, K. J. Cios and J. N. Clore, "Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records," BioMed Research International, vol. 2014, Article ID 781670, 2014.
[2]	Diabetes 130-US Hospitals for Years 1999–2008, UCI Machine Learning Repository, 2014. DOI: 10.24432/C5230J. Available: https://archive.ics.uci.edu/dataset/296
[3]	S. Masood, "Hospital Readmission Analysis," Kaggle notebook. Available: https://www.kaggle.com/code/sheemamasood/hospital-readmission-analysis
[4]	W. McKinney, "Data structures for statistical computing in Python," Proc. 9th Python in Science Conf., 2010. Pandas documentation: https://pandas.pydata.org/docs/
[5]	J. D. Hunter, "Matplotlib: A 2D graphics environment," Computing in Science & Engineering, vol. 9, no. 3, pp. 90–95, 2007.
[6]	M. L. Waskom, "seaborn: statistical data visualization," Journal of Open Source Software, vol. 6, no. 60, p. 3021, 2021.
[7]	P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," Nature Methods, vol. 17, pp. 261–272, 2020.
[8]	C. R. Harris et al., "Array programming with NumPy," Nature, vol. 585, pp. 357–362, 2020.
[9]	World Health Organization, open health data and health-system information. Available: https://www.who.int/data
Declaration of AI Tool Usage
In line with the course PBL guidelines, the team declares that an AI assistant (Claude, by Anthropic) was used to support code drafting and debugging, visualisation suggestions and documentation/formatting of this report. The team is responsible for the dataset selection, verification of the outputs, interpretation of results and the conclusions presented, and all numerical results were generated by running the Python code on the downloaded dataset.

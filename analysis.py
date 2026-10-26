import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
import json, warnings; warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep'); plt.rcParams['figure.dpi']=150
R={}
raw=pd.read_csv('diabetic_data.csv',na_values=['?'],low_memory=False)
R['raw_shape']=raw.shape
miss=(raw.isna().mean()*100); miss=miss[miss>0].sort_values(ascending=False); R['missing']=miss.round(2).to_dict()
R['dupes']=int(raw.duplicated().sum()); R['unique_patients']=int(raw.patient_nbr.nunique())
# ---- cleaning
df=raw.copy()
df=df.drop(columns=['weight','payer_code'])
const=[c for c in df.columns if df[c].nunique()==1]; R['const_cols']=const
df=df.drop(columns=const)
df['medical_specialty']=df['medical_specialty'].fillna('Unknown'); df['race']=df['race'].fillna('Unknown')
df['max_glu_serum']=df['max_glu_serum'].fillna('Not tested'); df['A1Cresult']=df['A1Cresult'].fillna('Not tested')
n0=len(df); df=df[df.gender!='Unknown/Invalid']; R['dropped_gender']=n0-len(df)
adm_type={1:'Emergency',2:'Urgent',3:'Elective',4:'Newborn',5:'Not available',6:'Not available',7:'Trauma center',8:'Not available'}
df['admission_type']=df.admission_type_id.map(adm_type)
src={1:'Physician referral',2:'Clinic referral',3:'HMO referral',4:'Transfer: hospital',5:'Transfer: SNF',6:'Transfer: other facility',7:'Emergency room',22:'Transfer within hospital',25:'Ambulatory surgery'}
df['admission_source']=df.admission_source_id.map(src).fillna('Other / unknown')
def dg(i):
    if i==1: return 'Home'
    if i in (6,8): return 'Home with health service'
    if i in (3,4,5,22,23,24,2,15,16,17,27,28,29,30): return 'Facility transfer / care'
    if i in (11,13,14,19,20,21): return 'Expired / hospice'
    if i==7: return 'Left against advice'
    return 'Other / unknown'
df['discharge_group']=df.discharge_disposition_id.map(dg)
def icd(x):
    if pd.isna(x): return 'Unknown'
    if str(x).startswith(('E','V')): return 'Other'
    v=float(x)
    if 390<=v<=459 or v==785: return 'Circulatory'
    if 460<=v<=519 or v==786: return 'Respiratory'
    if 520<=v<=579 or v==787: return 'Digestive'
    if int(v)==250: return 'Diabetes'
    if 800<=v<=999: return 'Injury'
    if 710<=v<=739: return 'Musculoskeletal'
    if 580<=v<=629 or v==788: return 'Genitourinary'
    if 140<=v<=239: return 'Neoplasms'
    return 'Other'
df['primary_diag_group']=df.diag_1.map(icd)
df['readmit_30']=(df.readmitted=='<30').astype(int)
age_order=sorted(df.age.unique(), key=lambda s:int(s.strip('[').split('-')[0]))
df['age']=pd.Categorical(df.age,categories=age_order,ordered=True)
df['age_mid']=df.age.map({a:int(a.strip('[').split('-')[0])+5 for a in age_order})
R['clean_shape']=df.shape
df.to_csv('cleaned_diabetic_data.csv',index=False)
R['expired_n']=int((df.discharge_group=='Expired / hospice').sum())
# analysis frame for readmission (exclude expired/hospice who cannot be readmitted)
rd=df[df.discharge_group!='Expired / hospice']; R['readm_n']=len(rd)
R['readm_rates']={k:round((rd.readmitted==k).mean()*100,2) for k in ['NO','>30','<30']}
R['readm_counts']=rd.readmitted.value_counts().to_dict()
R['los']=df.time_in_hospital.describe().round(3).to_dict(); R['los_median']=float(df.time_in_hospital.median())
R['los_mode']=int(df.time_in_hospital.mode()[0]); R['los_skew']=round(float(df.time_in_hospital.skew()),3)
def seg(col,frame=df):
    g=frame.groupby(col,observed=True).time_in_hospital
    t=pd.DataFrame({'n':g.size(),'mean':g.mean(),'sd':g.std(),'median':g.median()})
    t['ci95']=1.96*t.sd/np.sqrt(t.n); return t.round(3)
for c in ['age','admission_type','admission_source','discharge_group','primary_diag_group','gender']:
    R['los_by_'+c]=seg(c).reset_index().astype(str).to_dict('records')
    R['count_'+c]=df[c].value_counts().to_dict() if c!='age' else df[c].value_counts().sort_index().to_dict()
R['count_age']={str(k):int(v) for k,v in R['count_age'].items()}
def rate(col): 
    t=rd.groupby(col,observed=True).agg(n=('readmit_30','size'),rate=('readmit_30','mean')); t['rate']=(t.rate*100).round(2); return t.reset_index().astype(str).to_dict('records')
for c in ['age','admission_type','admission_source','discharge_group','primary_diag_group','gender']: R['r30_by_'+c]=rate(c)
rd2=rd.copy(); rd2['inp']=rd2.number_inpatient.clip(upper=4)
R['r30_by_inp']=rate_inp=rd2.groupby('inp').agg(n=('readmit_30','size'),rate=('readmit_30','mean')).assign(rate=lambda t:(t.rate*100).round(2)).reset_index().to_dict('records')
R['r30_overall']=round(rd.readmit_30.mean()*100,2)
# first-encounter sensitivity
R['first_enc_n']=int(df.patient_nbr.nunique()); R['multi_enc_patients']=int((df.patient_nbr.value_counts()>1).sum())
# ---- stats
S={}
groups=[g.time_in_hospital.values for _,g in df[df.admission_type.isin(['Emergency','Urgent','Elective'])].groupby('admission_type')]
S['anova_adm']=stats.f_oneway(*groups); S['kw_adm']=stats.kruskal(*groups)
groups=[g.time_in_hospital.values for _,g in df.groupby('age',observed=True)]; S['kw_age']=stats.kruskal(*groups)
groups=[g.time_in_hospital.values for _,g in df.groupby('discharge_group')]; S['kw_dis']=stats.kruskal(*groups)
a=rd[rd.readmit_30==1].time_in_hospital; b=rd[rd.readmit_30==0].time_in_hospital
S['mw_los_r30']=stats.mannwhitneyu(a,b); S['mean_los_r30']=(a.mean(),b.mean()); S['t_los_r30']=stats.ttest_ind(a,b,equal_var=False)
def chi(col):
    ct=pd.crosstab(rd[col],rd.readmit_30); c2,p,dof,_=stats.chi2_contingency(ct); n=ct.values.sum(); v=np.sqrt(c2/(n*(min(ct.shape)-1))); return (c2,p,dof,v)
for c in ['age','admission_type','discharge_group','gender','primary_diag_group']: S['chi_'+c]=chi(c)
num=['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses','age_mid']
S['corr']=df[num].corr().round(3)
S['spearman_med']=stats.spearmanr(df.time_in_hospital,df.num_medications); S['pearson_med']=stats.pearsonr(df.time_in_hospital,df.num_medications)
S['pearson_lab']=stats.pearsonr(df.time_in_hospital,df.num_lab_procedures)
S['pearson_dx']=stats.pearsonr(df.time_in_hospital,df.number_diagnoses)
S['pearson_age']=stats.pearsonr(df.time_in_hospital,df.age_mid)
S['pearson_inp_r']=stats.pearsonr(rd.number_inpatient,rd.readmit_30)
# mean LOS CI
m=df.time_in_hospital.mean(); se=df.time_in_hospital.std()/np.sqrt(len(df)); S['los_ci']=(m-1.96*se,m+1.96*se)
# outliers
q1,q3=df.time_in_hospital.quantile([.25,.75]); iqr=q3-q1; S['iqr_out']=int((df.time_in_hospital>q3+1.5*iqr).sum()); S['iqr_fence']=q3+1.5*iqr
def ser(o):
    if isinstance(o,pd.DataFrame): return o.to_dict()
    if hasattr(o,'__len__') and not isinstance(o,(tuple,str)) and len(o)==2 and hasattr(o,'__getitem__') and not hasattr(o,'statistic'): return [float(o[0]),float(o[1])]
    if hasattr(o,'statistic'): return [float(o.statistic),float(o.pvalue)]
    if hasattr(o,'correlation'): return [float(o.correlation),float(o.pvalue)]
    if isinstance(o,tuple): return [float(x) for x in o]
    return o
R['stats']={k:ser(v) for k,v in S.items()}
json.dump(R,open('results.json','w'),indent=1,default=str)
# ---- figures
def save(n): plt.tight_layout(); plt.savefig(f'figs/{n}.png'); plt.close()
plt.figure(figsize=(7,4)); miss.plot.barh(color='#c0504d'); plt.xlabel('% missing'); plt.title('Columns with missing values (raw data)'); plt.gca().invert_yaxis(); save('f1_missing')
plt.figure(figsize=(7,4)); sns.countplot(x='time_in_hospital',data=df,color='#4c72b0'); plt.axvline(df.time_in_hospital.mean()-1,color='r',ls='--',label=f'Mean = {df.time_in_hospital.mean():.2f} days'); plt.legend(); plt.xlabel('Length of stay (days)'); plt.ylabel('Encounters'); plt.title('Distribution of length of stay'); save('f2_los')
fig,ax=plt.subplots(1,2,figsize=(10,4)); 
vc=df.age.value_counts().sort_index(); ax[0].bar(range(len(vc)),vc.values,color='#4c72b0'); ax[0].set_xticks(range(len(vc))); ax[0].set_xticklabels(vc.index,rotation=60); ax[0].set_title('Admissions by age group'); ax[0].set_ylabel('Encounters')
sns.countplot(y='gender',data=df,ax=ax[1],color='#55a868'); ax[1].set_title('Admissions by gender'); save('f3_age_gender')
fig,ax=plt.subplots(1,2,figsize=(11,4))
vc=df.admission_type.value_counts(); sns.barplot(x=vc.values,y=vc.index,ax=ax[0],color='#4c72b0'); ax[0].set_title('Admissions by admission type'); ax[0].set_xlabel('Encounters')
vc=df.admission_source.value_counts(); sns.barplot(x=vc.values,y=vc.index,ax=ax[1],color='#dd8452'); ax[1].set_title('Admissions by source'); ax[1].set_xlabel('Encounters'); save('f4_adm')
plt.figure(figsize=(7,4)); vc=df.discharge_group.value_counts(); sns.barplot(x=vc.values,y=vc.index,color='#8172b3'); plt.title('Discharge disposition (grouped)'); plt.xlabel('Encounters'); save('f5_disch')
ct=pd.crosstab(df.admission_type,df.discharge_group,normalize='index')*100
plt.figure(figsize=(9,4)); sns.heatmap(ct.drop(index=['Newborn','Trauma center'],errors='ignore'),annot=True,fmt='.1f',cmap='YlGnBu'); plt.title('Admission type vs discharge destination (row %)'); plt.ylabel(''); plt.xlabel(''); save('f6_flow')
fig,ax=plt.subplots(1,2,figsize=(11,4))
sns.boxplot(x='age',y='time_in_hospital',data=df,ax=ax[0],color='#8fb4e3'); ax[0].tick_params(axis='x',rotation=60); ax[0].set_title('LOS by age group'); ax[0].set_ylabel('Days'); ax[0].set_xlabel('')
o=df[df.admission_type.isin(['Emergency','Urgent','Elective'])]; sns.boxplot(x='admission_type',y='time_in_hospital',data=o,ax=ax[1],color='#f2b28a'); ax[1].set_title('LOS by admission type'); ax[1].set_ylabel(''); ax[1].set_xlabel(''); save('f7_los_box')
plt.figure(figsize=(8,4)); t=seg('discharge_group'); t=t.sort_values('mean'); plt.barh(t.index,t['mean'],xerr=t.ci95,color='#8172b3'); plt.xlabel('Mean LOS (days) ± 95% CI'); plt.title('Mean length of stay by discharge destination'); save('f8_los_dis')
plt.figure(figsize=(8,4)); vc=df.primary_diag_group.value_counts(); sns.barplot(x=vc.values,y=vc.index,color='#c44e52'); plt.title('Primary diagnosis group'); plt.xlabel('Encounters'); save('f9_diag')
fig,ax=plt.subplots(1,3,figsize=(13,4))
rr=rd.groupby('age',observed=True).readmit_30.mean()*100; ax[0].plot(range(len(rr)),rr.values,marker='o'); ax[0].set_xticks(range(len(rr))); ax[0].set_xticklabels(rr.index,rotation=60); ax[0].set_title('30-day readmission % by age'); ax[0].set_ylabel('%')
rr=rd2.groupby('inp').readmit_30.mean()*100; ax[1].bar(['0','1','2','3','4+'],rr.values,color='#c44e52'); ax[1].set_title('by prior inpatient visits'); ax[1].set_xlabel('Prior inpatient visits')
rr=rd.groupby('discharge_group').readmit_30.mean()*100; rr=rr.sort_values(); ax[2].barh(rr.index,rr.values,color='#8172b3'); ax[2].set_title('by discharge destination'); save('f10_readm')
plt.figure(figsize=(7,5.5)); sns.heatmap(S['corr'],annot=True,fmt='.2f',cmap='coolwarm',center=0,annot_kws={'size':7}); plt.title('Correlation matrix (numeric attributes)'); save('f11_corr')
plt.figure(figsize=(6,4)); rd.readmitted.value_counts().reindex(['NO','>30','<30']).plot.bar(color=['#55a868','#dd8452','#c44e52']); plt.xticks(rotation=0); plt.title('Readmission outcome'); plt.ylabel('Encounters'); save('f12_readm')

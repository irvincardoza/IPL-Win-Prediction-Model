import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

match=pd.read_csv('matches.csv')
delivery=pd.read_csv('deliveries.csv')


total_score=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
# total_score=total_score+1
#print(total_score)

total_score=total_score[total_score['inning']==1]
#if = 2 we get 2nd innings total.



m=match.merge(total_score[['match_id','total_runs']],left_on='id',right_on='match_id')
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

#print(m['team1'].unique())
#this function prints all the teams of ipl, we need to filter the same teams with differemnt names.
if total_score.empty:
    print("hedjndnd")
m['team1']=m['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
m['team2']=m['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

m['team1']=m['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
m['team2']=m['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


#create a list of teams that we want to use


m=m[m['team1'].isin(teams)]
m=m[m['team2'].isin(teams)]

#remove mes affected by rain
#m = m[m['dl_applied'] == 0]

m=m[['match_id','city','winner','total_runs']]
d = m.merge(delivery,on='match_id')
d = d[d['inning'] == 2]


# #this gives us the total runs after each ball of the m
d['current_score'] = d.groupby('match_id')['total_runs_y'].cumsum()


d['runs_left']=d['total_runs_x']-d['current_score']

# #we now have runs left to win

d['balls_left']=126-(d['over']*6+d['ball'])
# #this formula gives us the balls left after each ball

#to find how many wickets left after each ball

d['player_dismissed']=d['player_dismissed'].fillna("0")
d['player_dismissed']=d['player_dismissed'].apply(lambda x:x if x=="0" else 1).astype(int)

#d['player_dismissed']=d['player_dismissed'].astype("int")
wickets=d.groupby('match_id')['player_dismissed'].cumsum().values
d['wickets']=10-wickets

#now we focus on crr, rrr and result
#crr=runs/over

d['crr'] = np.where(120 - d['balls_left'] > 0, (d['current_score'] * 6) / (120 - d['balls_left']), np.nan)
d['rrr'] = np.where(d['balls_left'] > 0, (d['runs_left'] * 6) / d['balls_left'], np.nan)

def result(row):
    return 1 if row['batting_team']==row['winner'] else 0
d['result']= d.apply(result,axis=1)

#these are the data feilds we require to calculate win probability
final = d[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]

final=final.sample(final.shape[0])
final.sample()

final.dropna(inplace=True)
final.dropna(subset=['crr', 'rrr'], inplace=True)
final = final[final['balls_left'] != 0]
#print(final.sample())

#now we train the data
X=final.iloc[:,:-1]
y=final.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#print(X_train.describe())

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])
pipe.fit(X_train,y_train)
y_predict=pipe.predict(X_test)

accuracy_score(y_test,y_predict)

def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target

# temp_df,target = match_progression(d,335986,pipe)
# print(temp_df)

###   FOR UNDERSTANDING USING GRAPHS   ###
# import matplotlib.pyplot as plt
# plt.figure(figsize=(18,8))
# plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
# plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
# plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
# plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
# plt.title('Target-' + str(target))

#we need to get cities to maken our web app
d['city'].unique()
pickle.dump(pipe,open('pipe.pkl','wb'))

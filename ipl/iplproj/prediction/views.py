import os
import pickle
from django.conf import settings
from django.shortcuts import render
import pandas as pd
pipe = pickle.load(open('./app/pipe.pkl', 'rb'))


def predict(request):
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
    cities = [
        'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
        'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
        'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
        'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
        'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
        'Sharjah', 'Mohali', 'Bengaluru'
    ]

    if request.method == 'POST':
        batting_team = request.POST.get('batting_team')
        bowling_team = request.POST.get('bowling_team')
        city = request.POST.get('city')
        target = int(request.POST.get('target'))
        score = int(request.POST.get('score'))
        overs = float(request.POST.get('overs'))
        wickets = int(request.POST.get('wickets'))

        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        model = pipe
        result = model.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        context = {
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'win_percentage': round(win * 100, 2),
            'loss_percentage': round(loss * 100, 2),
            'teams': teams,
            'cities': cities
        }

        return render(request, 'prediction/result.html', context)

    context = {
        'teams': teams,
        'cities': cities
    }
    return render(request, 'prediction/predict.html', context)

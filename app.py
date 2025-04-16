# Importing the necessary dependencies
import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Declaring the teams
teams = ['Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangaluru',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Lucknow Super Giants',
    'Gujarat Titans',
    'Delhi Capitals']

# Declaring the venues where the matches are going to take place
cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur',
          'Hyderabad', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban',
          'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Kochi', 'Indore', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi',
          'Abu Dhabi', 'Rajkot', 'Kanpur', 'Bengaluru', 'Dubai',
          'Sharjah', 'Navi Mumbai', 'Lucknow', 'Guwahati', 'Mohali']

# Loading our machine learning model from a saved pickle file
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Setting up the app's title
st.title('IPL Win Predictor')

# Setting up the layout with two columns
col1, col2 = st.columns(2)

# Creating a dropdown selector for the batting team
with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))

# Creating a dropdown selector for the bowling team
with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

# Creating a dropdown selector for the city where the match is being played
city = st.selectbox('Select the city where the match is being played', sorted(cities))

# Creating a numeric input for the target score
target = int(st.number_input('Target', step=1))

# Setting up the layout with three columns
col3, col4, col5 = st.columns(3)

# Current score
with col3:
    score = int(st.number_input('Score', step=1))

# Overs completed
with col4:
    overs = float(st.number_input('Overs Completed'))

# Wickets fallen
with col5:
    wickets_fallen = int(st.number_input('Wickets Fallen', step=1))

# Match result based on conditions
if score > target:
    st.write(battingteam, "won the match")

elif score == target - 1 and overs == 20:
    st.write("Match Drawn")

elif wickets_fallen == 10 and score < target - 1:
    st.write(bowlingteam, 'Won the match')

elif wickets_fallen == 10 and score == target - 1:
    st.write('Match tied')

elif battingteam == bowlingteam:
    st.write('To proceed, please select different teams because no match can be played between the same teams')

else:
    # Input validation
    if (
        0 <= target <= 300
        and 0 <= overs <= 20
        and 0 <= wickets_fallen <= 10
        and score >= 0
    ):
        try:
            if st.button('Predict Probability'):

                # Calculate derived values
                runs_left = target - score
                balls_left = 120 - int(overs * 6)
                wickets = 10 - wickets_fallen
                currentrunrate = score / overs if overs > 0 else 0
                requiredrunrate = (runs_left * 6 / balls_left) if balls_left > 0 else 0

                # Build DataFrame for model input
                input_df = pd.DataFrame({
                    'batting_team': [battingteam],
                    'bowling_team': [bowlingteam],
                    'city': [city],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets_left': [wickets],
                    'total_runs_x': [target],
                    'cur_run_rate': [currentrunrate],
                    'req_run_rate': [requiredrunrate]
                })

                # Predict probabilities
                result = pipe.predict_proba(input_df)
                lossprob = result[0][0]
                winprob = result[0][1]

                # Display results
                st.header(battingteam + " - " + str(round(winprob * 100)) + "%")
                st.header(bowlingteam + " - " + str(round(lossprob * 100)) + "%")

        except ZeroDivisionError:
            st.error("Please fill all the details")

    else:
        st.error("There is something wrong with the input, please fill the correct details")

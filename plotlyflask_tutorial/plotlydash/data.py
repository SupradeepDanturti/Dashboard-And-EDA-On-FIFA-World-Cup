"""Prepare data for Plotly Dash."""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.offline as iplot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output# Load Data
import seaborn as sns
import cufflinks as cf
import base64

iplot.init_notebook_mode(connected=True)
cf.go_offline()

cf.getThemes()
cf.set_config_file(theme = 'white')

def create_dataframe():
    """Create Pandas DataFrame from local CSV."""
    df = pd.read_csv('data/WorldCups.csv')
    return df

def all_cleaned_data():
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Importing data--------------------------------------
    # -----------------------------------------------------------------------

    # worldcup_df = pd.read_csv('/gdrive/My Drive/WorldCups.csv')
    # players_df = pd.read_csv('/gdrive/My Drive/WorldCupPlayers.csv')
    # matches_df = pd.read_csv('/gdrive/My Drive/WorldCupMatches.csv')

    worldcup_df = pd.read_csv ('data/WorldCups.csv')
    players_df = pd.read_csv ('data/WorldCupPlayers.csv')
    matches_df = pd.read_csv ('data/WorldCupMatches.csv')

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-1---------------------------------------------
    # -------------Maximum times a team appeared in top 3 positions----------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    matches_df.dropna (inplace = True)
    matches_df['Home Team Name'].value_counts ()
    matches_df[matches_df['Home Team Name'].str.contains ('rn')]
    names_to_be_corrected = matches_df[matches_df['Home Team Name'].str.contains ('rn"')][
        'Home Team Name'].value_counts ()
    wrong_list_of_country_names = list (names_to_be_corrected.index)
    corrected_list_of_names = [names_to_be_corrected.split ('>')[1] for names_to_be_corrected in
                               wrong_list_of_country_names]
    for index, wr in enumerate (wrong_list_of_country_names):
        worldcup_df = worldcup_df.replace (wrong_list_of_country_names[index], corrected_list_of_names[index])
    for index, wr in enumerate (wrong_list_of_country_names):
        players_df = players_df.replace (wrong_list_of_country_names[index], corrected_list_of_names[index])
    for index, wr in enumerate (wrong_list_of_country_names):
        matches_df = matches_df.replace (wrong_list_of_country_names[index], corrected_list_of_names[index])
    names_to_be_corrected = matches_df[matches_df['Home Team Name'].str.contains ('rn"')][
        'Home Team Name'].value_counts ()
    other_set_of_wrong_country_names = ['Germany FR', "Cï¿½te d'Ivoire", 'German DR', 'IR Iran']
    corrected_other_set_of_wrong_country_names = ['Germany', "Côte d'Ivoire", 'Germany', 'Iran']
    for index, wr in enumerate (other_set_of_wrong_country_names):
        worldcup_df = worldcup_df.replace (other_set_of_wrong_country_names[index],
                                           corrected_other_set_of_wrong_country_names[index])
    for index, wr in enumerate (other_set_of_wrong_country_names):
        matches_df = matches_df.replace (other_set_of_wrong_country_names[index],
                                         corrected_other_set_of_wrong_country_names[index])
    for index, wr in enumerate (other_set_of_wrong_country_names):
        players_df = players_df.replace (other_set_of_wrong_country_names[index],
                                         corrected_other_set_of_wrong_country_names[index])
    winners_worldcup = worldcup_df['Winner'].value_counts ()
    runners_worldcup = worldcup_df['Runners-Up'].value_counts ()
    third_worldcup = worldcup_df['Third'].value_counts ()
    all_team_details = pd.concat ([winners_worldcup, runners_worldcup, third_worldcup], axis = 1)
    all_team_details.fillna (0, inplace = True)
    fig_1 = all_team_details.iplot (kind = 'bar', xTitle = 'Teams', yTitle = 'Count',
                                    title = 'Maximum times a team appeared in top 3 positions', asFigure = True)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-2---------------------------------------------
    # --------------Number of goals per team---------------------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    home_goals = matches_df[['Home Team Name', 'Home Team Goals']]
    away_goals = matches_df[['Away Team Name', 'Away Team Goals']]
    home_goals.columns = ['Countries', 'Goals']
    away_goals.columns = ['Countries', 'Goals']
    total_goals = home_goals.append (away_goals, ignore_index = True)
    total_goals = total_goals.groupby ('Countries').sum ()
    total_goals = total_goals.sort_values (by = 'Goals', ascending = False)
    final_plot_graph = total_goals[:30]
    fig_2 = final_plot_graph.iplot (kind = 'bar', xTitle = 'Country Names', yTitle = 'No. of goals',
                                    title = 'Number of goals per team', asFigure = True)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-3---------------------------------------------
    # ---------------Teams with most goals per World Cup----------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    home_goals_df = matches_df.groupby (['Year', 'Home Team Name'])['Home Team Goals'].sum ()
    away_goal_df = matches_df.groupby (['Year', 'Away Team Name'])['Away Team Goals'].sum ()
    concat_goals_df = pd.concat ([home_goals_df, away_goal_df], axis = 1)
    concat_goals_df.fillna (0, inplace = True)
    concat_goals_df['Goals'] = concat_goals_df['Home Team Goals'] + concat_goals_df['Away Team Goals']
    df_of_goals_per_country_per_year = concat_goals_df.drop (['Home Team Goals', 'Away Team Goals'], axis = 1)
    df_of_goals_per_country_per_year = df_of_goals_per_country_per_year.reset_index ()
    df_of_goals_per_country_per_year.columns = ['Year', 'Country', 'Goals']
    df_of_goals_per_country_per_year = df_of_goals_per_country_per_year.sort_values (by = ['Year', 'Goals', ],
                                                                                     ascending = [True, False])
    top3 = df_of_goals_per_country_per_year.groupby ('Year').head (3)
    X = df_of_goals_per_country_per_year['Year'].values
    Y = df_of_goals_per_country_per_year['Goals'].values
    for teams in top3['Country'].drop_duplicates ().values:
        teams
    all_values = []
    for teams in top3['Country'].drop_duplicates ().values:
        year = top3[top3['Country'] == teams]['Year']
        goal = top3[top3['Country'] == teams]['Goals']
        all_values.append (go.Bar (x = year, y = goal, name = teams))
        layout = go.Layout (barmode = 'stack', title = 'Teams with most goals per World Cup')

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-4---------------------------------------------
    # ---------------Attendance per World-Cup--------------------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    worldcup_df['Attendance'] = worldcup_df['Attendance'].str.replace ('.', '')
    # graph_of_attendance = sns.barplot(x='Year',y='Attendance',data=worldcup_df)
    # graph_of_attendance.set_title('Attendance per World-Cup')

    # img

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-5---------------------------------------------
    # --------------Goals Scored per world-Cup-------------------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # graph_of_qt = sns.barplot(x='Year',y='GoalsScored',data=worldcup_df)
    # graph_of_qt.set_title('Goals Scored per world-Cup')
    # img

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-6---------------------------------------------
    # -------------Teams Qualified per World Cup-----------------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # graph_of_qt = sns.barplot(x='Year',y='MatchesPlayed',data=worldcup_df)
    # graph_of_qt.set_title(' Teams Qualified per World Cup')

    # img

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-7---------------------------------------------
    # ------------Matches with highest number of Attendance------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # matches_df['Datetime'] = pd.to_datetime(matches_df['Datetime'])
    # matches_df['Datetime'] = matches_df['Datetime'].apply(lambda x:x.strftime('%d %b, %Y'))
    # top5_matches = matches_df.sort_values(by ='Attendance', ascending = False)[:5]

    # top5_matches['vs'] = top5_matches['Home Team Name'] + ' vs ' + top5_matches['Away Team Name']
    # plt.figure(figsize=(12,9))

    # top5_matches_graph = sns.barplot(y = top5_matches['vs'], x = top5_matches['Attendance'])
    # sns.despine(right= True)

    # plt.ylabel("Match Teams")
    # plt.xlabel("Attendance")
    # plt.title("Matches with highest number of Attendance")

    # for i, s in enumerate ("Stadium: "+ top5_matches['Stadium']+ " Date: " + top5_matches['Datetime']):
    #    top5_matches_graph.text(1500, i, s, fontsize = 14)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-8---------------------------------------------
    # ---------------Matches with Least number of Attendance-----------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # bottom5_matches = matches_df.sort_values(by ='Attendance')[:5]
    # bottom5_matches['vs'] = bottom5_matches['Home Team Name'] +' vs ' + bottom5_matches['Away Team Name']
    # plt.figure(figsize=(12,9))

    # bottom5_matches_graph = sns.barplot(y = bottom5_matches['vs'], x = bottom5_matches['Attendance'])
    # sns.despine(right= True)

    # plt.ylabel("Match Teams")
    # plt.xlabel("Attendance")
    # plt.title("Matches with Least number of Attendance")

    # for i, s in enumerate ("Stadium: "+ bottom5_matches['Stadium']+ " Date: " + bottom5_matches['Datetime']):
    #    bottom5_matches_graph.text(375, i, s, fontsize = 12)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-9---------------------------------------------
    # -------------Stadium with highest avg attendance ----------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    matches_df['Year'] = matches_df['Year'].astype (int)
    stadium_with_high_avg_attendance_df = matches_df.groupby (['Stadium', 'City'])[
        'Attendance'].mean ().reset_index ().sort_values (by = 'Attendance', ascending = False)
    top10_stadium_with_highest_avg_attendance = stadium_with_high_avg_attendance_df[:10]
    # top10_stadium_with_highest_avg_attendance_fig=px.bar(top10_stadium_with_highest_avg_attendance,x='Stadium', y='Attendance')
    # top10_stadium_with_highest_avg_attendance_fig.update_layout(title="Stadium with highest avg attendance")
    # Despine top10_stadium_with_highest_avg_attendance.spines['right'].set_visible(False)
    # fig_9 = top10_stadium_with_highest_avg_attendance_fig.show()

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-10--------------------------------------------
    # -------------Cities With high Avg Attendance---------------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    fig_10 = matches_df['City'].value_counts ()[:20].iplot (kind = 'bar', title = 'Cities With high Avg Attendance',
                                                            asFigure = True)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # -------------------Graph-11--------------------------------------------
    # -------------Stadium with highest avg attendance ----------------------
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    df_of_groups_attendance = matches_df[['Stage', 'Attendance']]
    old_group_names = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6', 'Group A', 'Group B',
                       'Group C', 'Group D', 'Group E', 'Group F', 'Group G', 'Group H', 'Preliminary round',
                       'Round of 16', 'First round ', 'Preliminary round', 'First round']
    new_group_names = ['Group stage', 'Group stage', 'Group stage', 'Group stage', 'Group stage', 'Group stage',
                       'Group stage', 'Group stage', 'Group stage', 'Group stage', 'Group stage', 'Group stage',
                       'Group stage', 'Group stage', 'Group stage', 'Group stage', 'Group stage', 'Group stage',
                       'Group stage']
    for index, wr in enumerate (old_group_names):
        df_of_groups_attendance = df_of_groups_attendance.replace (old_group_names[index], new_group_names[index])
    old_thirdplace_names = ['Match for third place', 'Third place', 'Play-off for third place']
    new_thirdplace_names = ['Third place', 'Third place', 'Third place']
    for index, wr in enumerate (old_thirdplace_names):
        df_of_groups_attendance = df_of_groups_attendance.replace (old_thirdplace_names[index],
                                                                   new_thirdplace_names[index])
    old_playoffs_names = ['Quarter-finals', 'Semi-finals', 'Third place']
    new_playoffs_names = ['Playoffs', 'Playoffs', 'Playoffs']
    for index, wr in enumerate (old_playoffs_names):
        df_of_groups_attendance = df_of_groups_attendance.replace (old_playoffs_names[index], new_playoffs_names[index])

    final_df_for_attendance_of_group_stage = df_of_groups_attendance.groupby ('Stage').mean ()
    final_df_for_attendance_of_group_stage.reset_index (inplace = True)

    # fig_for_attendance_stage = px.pie(final_df_for_attendance_of_group_stage,names='Stage',values='Attendance', title='Average attendance of playoffs and Group stage matches')
    # fig_11= fig_for_attendance_stage.show()
    return all_cleaned_data




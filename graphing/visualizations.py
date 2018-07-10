"""
Stores generalized visualization functions that can be used to create graphs for 
exploratory analysis, blogs, briefs, presentations, or other content.
"""


__author__ = "Austin Herrick"
__copyright__ = "Copyright 2018, Penn Wharton Budget Model"


import warnings
import json
import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW as stats
from IPython.display import HTML
import matplotlib.dates as mdates

plt.style.use(['classic', 'pwbm.mplstyle')])


def graphing_ready_dataframe(
    df, 
    demographic, 
    interest_var,
    moment_type='Frequency',
    interest_value=1, 
    weights=None, 
    convert_to_annual=False,
    datetime=False,
    datetime_format='%Y'
):
    '''
    Given a dataframe, and metrics along which to measure, prepares information for graphing
    
    - df: The dataframe containing the data to be analyzed
    - demographic: The independent variable of interest
    - interest_var: The dependent variable of interest
    - moment_type: The type of comparison between dependent/independent variables. 'Frequency' compares
        how often the dependent var takes a particular value (default 1), 'Mean' compares the average value
        of the dependent variable by each independent dimension
    - interest_value: The value of interest for the dependent variable
    - weights: The variable used for weighting of the dataframe, if necessary
    - convert_to_annual: When enabled, converts probabilities associated with monthly transitions to annual
    - datetime: When enabled (if the demographic of interest is time-based), converts the results dataframe
        to datetime
    - datetime_format: Controls the format of the datetime conversion. Defaults to presenting data as years
    '''
    
    # run assertions and warnings for misspecified inputs
    moment_types = ['Frequency', 'Mean']
    assert (moment_type in moment_types), 'Invalid Moment Type! Please choose one of {}'.format(moment_types)
    assert (df[interest_var].dtype == 'float64' or df[interest_var].dtype == 'int64'), \
        'Dependent variable is non-numeric! Please convert dependent variable to an integer or float!'
    if weights:
        if len(df[df[weights].isnull()]) > 0:
            warnings.warn(
                'Warning: Weight variable contains nulls! Null values have been filled with 0 via "DummyWeight"'
            )
            df['DummyWeight'] = df[weights].fillna(0)
            weights = 'DummyWeight'
    
    # if no weight is provided, use a set of dummy weights
    if not weights:
        df['DummyWeight'] = 1
        weights = 'DummyWeight'
    
    # create storage structure for results dataframe
    results_list = []
    labels = ['Value', 'Moment', 'StandardError']
    
    # collect possible values taken along the chosen dimension
    values = sorted(df[demographic].unique())
    
    # build graphing dataframe
    for value in values:
        sub = df[df[demographic] == value]
        
        # find moment and standard error for variable of interest within the demographic subset
        if moment_type == 'Frequency': 
            moment = sub[sub[interest_var] == interest_value][weights].sum() / sub[weights].sum()
            statistics = stats(sub[interest_var], weights = sub[weights])
        elif moment_type == 'Mean':
            statistics = stats(sub[interest_var], weights = sub[weights])
            moment = statistics.mean
            
        standard_error = statistics.std / np.sqrt(len(sub))
            
        # convert monthly transitions to annual (if applicable)
        if convert_to_annual:
            moment = 1 - ((1 - moment)**12)
            standard_error = 1 - ((1 - standard_error)**12)
        
        # append to results
        results_list.append([value, moment, standard_error])
        
    # create dataframe
    results = pd.DataFrame.from_records(results_list, columns=labels)
    
    # if necessary, convert to datetime
    if datetime:
        results['Value'] = pd.to_datetime(results.Value, format=datetime_format)
        
    return results

def single_visualization(
    results,
    demographic,
    interest_var,
    moment_type='Frequency',
    categorical=False,
    categorical_coding=None,
    custom_title=None,
    custom_axis=None,
    subtitle=None,
    max_line_length=80
):
    '''
    Visualize a function. Create bar/line graphs for one data series, without comparison to another
    
    - results: A dataframe containing the information to be graphed
    - demographic: The independent variable of interest
    - interest_var: The dependent variable of interest
    - moment_type: The type of comparison contained within the results dataframe (used for labelling purposes)
    - categorical: Controls whether the graph displays a line or bar chart
    - categorical_coding: Allows users to control the order in which categorical information is displayed.
        Inputs are of the form of a list of all categorical keys, in the order the user would like them to
        appear.
    - custom_title: Allows users to submit custom titles, rather than using the default code-generated title
    - custom_axis: Allows users to submit custom y-axis labels, rather than using the default
    - subtitle: Allows users to submit text for a subtitle
    - max_line_length: Changes the default line length before a line break is created for subtitle formatting
    '''
    
    # run assertions and warnings for misspecified inputs
    if categorical:
        if not categorical_coding:
            warnings.warn(
                'Warning: Categorical call includes no coding list, default ordering will be used. To \
                specify a coding list, supply a list of each category in the order you want them to \
                appear in your graph'
            )
            categorical_coding = results.Value.unique().tolist()
        else:
            assert(set(categorical_coding) == set(results.Value.unique().tolist())), \
                'Categorical codings do not match values in result dataframe! Compare supplied codings: \
                {} to values in dataframe: {}'.format(
                    set(categorical_coding), 
                    set(results.Value.unique().tolist())
                )

    # set boundary conditions
    width = 0.7
    x_basis = 0
    y_lim_min = 0
    y_lim_max = results.Moment.max() * 1.2
    graph_height = y_lim_max - y_lim_min
    lower_footer = 0
    
    # controls boundaries when line graphs are displayed (important for datetime functionality)
    if categorical:
        x_adjuster = 0
    else:
        x_adjuster = results.Value.min()
    
    # convert to list of desired values
    moments = results.Moment.values.tolist()
    std_errors = results.StandardError.values.tolist()
    
    f, ax = plt.subplots(1)
    
    if categorical:
        
        # graph cateogrical information, using a specified order and containing error bars
        labels = categorical_coding
        ind = np.arange(len(results))
        plot = ax.bar(ind + 0.2, moments, width, yerr = std_errors)
        ax.set_xlabel("{}".format(demographic), fontsize = 12)
        ax.set_xticks(ind + 0.2 + width/2)
        ax.set_xticklabels(labels, fontsize = 9)
        
        # rotate labels for categories that are too long or have too many items
        if len(labels) > 10:
            plt.xticks(rotation = 45)
            max_label_length = len(max(labels, key = len))
            lower_footer = 0.0066 * max_label_length
            
    else:
        plot = plt.plot(results.Value, results.Moment, label = '{}'.format(demographic))
        
        # adds confidence interval bars to line graphs
        plt.plot(
            results.Value,
            results.Moment + (results.StandardError * 1.96),
            label = '{}'.format(demographic),
            linestyle = ':',
            color = '#004785',
            linewidth = 2.4
        )
        plt.plot(
            results.Value,
            results.Moment - (results.StandardError * 1.96),
            label = '{}'.format(demographic),
            linestyle = ':',
            color = '#004785',
            linewidth = 2.4
        )
        
    # handle y-axis formatting
    ax.set_ylim(y_lim_min, y_lim_max)
    if custom_axis: 
        ax.set_ylabel(custom_axis, fontsize = 14)
    else:
        ax.set_ylabel('{} of {}'.format(moment_type, interest_var), fontsize = 14)
    

    # add subtitle
    line_count = 0
    if subtitle:
        # parse user-submitted subtitle, adding line breaks as needed
        parsed_subtitle, line_count = parse_subtitle(subtitle, max_line_length)
        ax.text(
            x = x_adjuster,
            y = y_lim_max,
            s = parsed_subtitle,
            fontsize = 18,
            alpha = 0.5
        )
        
    # add title
    if custom_title:
        title = custom_title
    else:
        title = "{} frequency by {}".format(interest_var, demographic)  
    
    ax.text(
        x = x_adjuster,
        y = y_lim_max + (graph_height * (0.05 * line_count)),
        s = title,
        fontsize = 18,
        weight = 'bold'
    )
    
    # display figure
    plt.rcParams['figure.figsize'] = [13.5, 7.5]
    plt.show()


def parse_subtitle(subtitle, max_line_length = 80):
    '''
    Uses string comprehension to add linebreaks into user-submitted subtitles.
    
    - subtitle: A string containing the text to be used as the subtitle
    - max_line_length: An integer specifying the maximum line length
    '''
    
    # split the submitted subtitle by individual words, and initialize the parsed subtitle
    word_list = subtitle.split()
    parsed_text = word_list[0]
    original_line_length = max_line_length
    
    # verify that no single word exceeds the max line length
    longest_word = max(word_list, key=len)
    if len(longest_word) > max_line_length:
        max_line_length = len(longest_word)
        warnings.warn('Single word "{}" exceeds default maximum line length, extending max length to {}'.format(longest_word, max_line_length))
    
    # for each word, add it to the parsed subtitle unless it would create a line longer than the cutoff
    for word in word_list[1:]:
        tentative = parsed_text + ' ' + word
        if len(tentative) < max_line_length:
            parsed_text = tentative
            
        # when a line is too long, create a newline character and increment the cutoff
        else:
            max_line_length += original_line_length
            parsed_text = parsed_text + '\n' + word
            
    # count the number of lines in the subtitle, in order to ensure proper title spacing
    line_count = len(re.findall('\n', parsed_text)) + 1
            
    return parsed_text, line_count
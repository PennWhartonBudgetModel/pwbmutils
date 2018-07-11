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
			try:
				moment = sub[sub[interest_var] == interest_value][weights].sum() / sub[weights].sum()
				statistics = stats(sub[interest_var], weights = sub[weights])
			except ZeroDivisionError:
				moment = 0
				statistics = 0
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


def visualization(
	result_list,
	demographic,
	interest_var,
	label_list=None,
	moment_type='Frequency',
	categorical=False,
	categorical_coding=None,
	custom_title=None,
	custom_axis=None,
	subtitle=None,
	max_line_length=80,
	save_location=None,
	legend_location=None
):
	'''
	Visualize a function. Create bar/line graphs for one data series, without comparison to another
	
	- result_list: A list containing each dataframe to be graphed
	- demographic: The independent variable of interest
	- interest_var: The dependent variable of interest
	- label_list: Labels used for legend creation
	- moment_type: The type of comparison contained within the results dataframe (used for labelling purposes)
	- categorical: Controls whether the graph displays a line or bar chart
	- categorical_coding: Allows users to control the order in which categorical information is displayed.
		Inputs are of the form of a list of all categorical keys, in the order the user would like them to
		appear.
	- custom_title: Allows users to submit custom titles, rather than using the default code-generated title
	- custom_axis: Allows users to submit custom y-axis labels, rather than using the default
	- subtitle: Allows users to submit text for a subtitle
	- max_line_length: Changes the default line length before a line break is created for subtitle formatting
	- save_location: When enabled, saves the created graph to a save location
	- legend_location: The position on the graph to place the legend
	'''
	
	# if the user submits a single dataframe, converts to a list for standardized formatting
	if type(result_list) == pd.core.frame.DataFrame:
		result_list = [result_list]
	
	# run assertions and warnings for misspecified inputs
	if categorical:
		if not categorical_coding:
			warnings.warn(
				'Warning: Categorical call includes no coding list, default ordering will be used. To \
				specify a coding list, supply a list of each category in the order you want them to \
				appear in your graph'
			)
			categorical_coding = result_list[0].Value.unique().tolist()
		else:
			assert(set(categorical_coding) == set(result_list[0].Value.unique().tolist())), \
				'Categorical codings do not match values in result dataframe! Compare supplied codings: \
				{} to values in dataframe: {}'.format(
					set(categorical_coding), 
					set(result_list[0].Value.unique().tolist())
				)
				
	if len(result_list) > 1:
		if not label_list:
			warnings.warn('Warning: No labels were provided! Legend will use default naming instead. Provide \
			labels by submitting a list of strings for the legend')
			
			# construct default label list
			label_list = []
			for i in range(len(result_list)):
				label_list.append('Data {}'.format(i))

	
	f, ax = plt.subplots(1)
	
	if categorical:
		graph_categorical(ax, categorical_coding, result_list, demographic, label_list)
	else:
		graph_non_categorical(result_list, demographic, legend_location, label_list)
	
	# add title/subtitle
	add_labels(
		ax, 
		result_list,
		categorical,
		moment_type, 
		subtitle, 
		max_line_length, 
		interest_var, 
		demographic, 
		custom_title,
		custom_axis
	)

	# control frame size
	plt.rcParams['figure.figsize'] = [13.5, 7.5]
	
	# save figure, if necessay
	if save_location:
		savefig(save_location)
		
	# display figure
	plt.show()

def graph_categorical(ax, categorical_coding, result_list, demographic, label_list):
	'''
	Create plot for categorical graphs
	'''

	# graph cateogrical information, using a specified order and containing error bars
	width = 0.7 / len(result_list)
	labels = categorical_coding
	ind = np.arange(len(result_list[0]))
	
	# sequentially plot resuls from each dataframe
	datasets = []
	for i in range(len(result_list)):
		moments = result_list[i].Moment.values.tolist()
		std_errors = result_list[i].StandardError.values.tolist()
		plot = ax.bar(ind + 0.2 + (width * i), moments, width, yerr = std_errors)
		
		# create a holding list of all graphed data, to be used for legend creation
		datasets.append(plot[0])
		
	ax.set_xlabel("{}".format(demographic), fontsize = 12)
	ax.set_xticks(ind + 0.2 + 0.35)
	ax.set_xticklabels(labels, fontsize = 9)
	
	# rotate labels for categories that are too long or have too many items
	if len(labels) > 10:
		plt.xticks(rotation = 45)
		max_label_length = len(max(labels, key = len))
		
	# add legend, if multiple lines are being graphed
	if len(result_list) > 1:
		leg = plt.legend(
			tuple(datasets),
			label_list,
			bbox_to_anchor = (-0.03, 1),
			loc = 2,
			fontsize = 9,
			ncol = len(result_list)
		)

def graph_non_categorical(result_list, demographic, legend_location, label_list):
	'''
	Create plot for non-categorical graphs
	'''

	for dataframe in result_list:
		plot = plt.plot(dataframe.Value, dataframe.Moment, label = '{}'.format(demographic))

	# retrieve color cycler to match standard error bars to the original line's color scheme
	color_cylcer = plt.rcParams['axes.prop_cycle'].by_key()['color']
	
	for i in range(len(result_list)):  
		# adds confidence interval bars to line graphs
		plt.plot(
			result_list[i].Value,
			result_list[i].Moment + (result_list[i].StandardError * 1.96),
			label = '{}'.format(demographic),
			linestyle = ':',
			color = color_cylcer[i],
			linewidth = 2.4
		)
		plt.plot(
			result_list[i].Value,
			result_list[i].Moment - (result_list[i].StandardError * 1.96),
			label = '{}'.format(demographic),
			linestyle = ':',
			color = color_cylcer[i],
			linewidth = 2.4
		)
	
	# add legend
	if len(result_list) > 1:
		if legend_location:
			leg = plt.legend(
				label_list, 
				loc = legend_location,
				fontsize = 9,
				ncol = len(result_list)
			)
		else:
			leg = plt.legend(
				label_list, 
				bbox_to_anchor = (-0.03, 1),
				loc = 2,
				fontsize = 9,
				ncol = len(result_list)
			)

def add_labels(
		ax,
		result_list,
		categorical,
		moment_type, 
		subtitle, 
		max_line_length, 
		interest_var, 
		demographic, 
		custom_title,
		custom_axis
	):
	'''
	Add title, subtitle, and y-axis labelling to the graph
	'''

	# set boundary conditions
	y_lim_min = 0
	# adjust height of graph relative to largest number to be graphed
	y_lim_max = max([i.Moment.max() for i in result_list]) * 1.2
	graph_height = y_lim_max - y_lim_min
	
	# controls boundaries when line graphs are displayed (important for datetime functionality)
	if categorical:
		x_adjuster = 0
	else:
		x_adjuster = min([i.Value.min() for i in result_list])

	# handle y-axis formatting
	ax.set_ylim(y_lim_min, y_lim_max)
	if custom_axis: 
		ax.set_ylabel(custom_axis, fontsize = 14)
	else:
		ax.set_ylabel('{} of {}'.format(moment_type, interest_var), fontsize = 14)

	# add subtitle
	if subtitle:
		# parse user-submitted subtitle, adding line breaks as needed
		parsed_subtitle, line_count = parse_subtitle(subtitle, max_line_length)
		ax.text(
			x = x_adjuster,
			y = y_lim_max + (graph_height * 0.05),
			s = parsed_subtitle,
			fontsize = 15,
			alpha = 0.5
		)
	else:
		line_count = 0 

	# add title
	if custom_title:
		title = custom_title
	else:
		title = "{} frequency by {}".format(interest_var, demographic)  
	
	ax.text(
		x = x_adjuster,
		y = y_lim_max + (graph_height * ((0.04 * line_count) + 0.07)),
		s = title,
		fontsize = 18,
		weight = 'bold'
	)


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
"""
Stores generalized visualization functions that can be used to create graphs for 
exploratory analysis, blogs, briefs, presentations, or other content.
"""

__author__ = "Austin Herrick"
__copyright__ = "Copyright 2018, Penn Wharton Budget Model"

import warnings
import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW as stats
from pandas.api.types import is_numeric_dtype

import graphing.utilities

def graphing_ready_dataframe(
	df, 
	demographic, 
	interest_var,
	moment_type='Mean',
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
	assert (is_numeric_dtype(df[interest_var])), \
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
	try:
		values = sorted(df[demographic].unique())
	except TypeError:
		values = df[demographic].unique()
	
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

	# load the style guide
	plt.style.use(['classic', 'pwbm'])
	
	# if the user submits a single dataframe, converts to a list for standardized formatting
	if type(result_list) == pd.core.frame.DataFrame:
		result_list = [result_list]
	
	# run assertions and warnings for misspecified inputs
	if categorical:
		if not categorical_coding:
			warnings.warn(
				'Warning: Categorical call includes no coding list, default ordering will be used. To '
				'specify a coding list, supply a list of each category in the order you want them to '
				'appear in your graph'
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
			warnings.warn('Warning: No labels were provided! Legend will use default naming instead. Provide '
			'labels by submitting a list of strings for the legend')
			
			# construct default label list
			label_list = []
			for i in range(len(result_list)):
				label_list.append('Data {}'.format(i))

	
	f, ax = plt.subplots(1)
	
	if categorical:
		graphing.utilities.graph_categorical(ax, categorical_coding, result_list, demographic, label_list)
	else:
		graphing.utilities.graph_non_categorical(result_list, demographic, legend_location, label_list)
	
	# add title/subtitle
	graphing.utilities.add_labels(
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


def sample_dataset(df, column='HouseholdID', max_count=1000):
	'''
	Samples a subset a dataset, to create an abridged version

	- df: The dataframe to abridge
	- column: The group from which entries are chosen.
	- max_count: The number of entries to draw
	'''

	units = df[column].unique().tolist()
	count = 0
	sample_choices = []

	while count < max_count:
		
		if count % 100 == 0:
			print('Sampling {}th entry....'.format(count))

		choice = np.random.choice(range(len(units)))
		sample_choices.append(units[choice])
		count += 1
		del units[choice]

	sample = df[df[column].isin(sample_choices)]

	return sample

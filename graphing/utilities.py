'''
Contains helper functions used during visualization
'''

import warnings
import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW as stats

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
		title = "{} {} by {}".format(interest_var, moment_type, demographic)  
	
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
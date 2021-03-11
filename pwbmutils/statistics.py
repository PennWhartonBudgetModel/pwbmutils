"""Collection of convenience functions for statistical work.
"""

__author__ = 'Nick Janetos, Alex Arnon, Austin Herrick'
__copyright__ = '2021 Penn Wharton Budget Model'

# pylint: disable=E1101, C0103

from copy import copy
import math
import pickle
import re

from numba import njit
import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
from statsmodels.regression.linear_model import WLS
np.warnings.filterwarnings("ignore")
from statsmodels.api import GLM
np.warnings.resetwarnings()
from statsmodels.genmod.families.family import Binomial
from statsmodels.genmod.families.links import logit
from statsmodels.formula.api import mnlogit
from patsy import ContrastMatrix


@njit
def logit_transform(X, betas):
	"""Performs a logistic transformation on data.

	Arguments:
		X {np.array} -- nxm np array.
		betas {np.array} -- mx1 np array with coefficients.

	Returns:
		np.array -- nx1 np array with logistic transformation applied.
	"""

	return 1 / (1 + np.exp(-1 * X @ betas))


@njit
def linear_transform(X, betas):
	"""Performs a linear transformation on data.

	Arguments:
		X {np.array} -- nxm np array.
		betas {np.array} -- mx1 np array with coefficients.

	Returns:
		np.array -- nx1 np array with linear transformation applied.
	"""

	return X @ betas


def inverse_mills_ratio(x, mu=0, sigma=1, invert=False):
	"""Computes the inverse mills ratio of some data.

	Arguments:
		x {np.array} -- nx1 np array of data.

	Keyword Arguments:
		mu {int} -- The mean of the normal distribution. (default: {0})
		sigma {int} -- The standard deviation of the normal distribution. (default: {1})
		invert {bool} -- Which of two sides to compute, defaults to <=. (default: {False})

	Returns:
		np.array -- nx1 np array of transformed data
	"""

	if not invert:
		return norm.pdf((x - mu) / sigma) / (1 - norm.cdf((x - mu) / sigma))
	else:
		return -1 * norm.pdf((x - mu) / sigma) / (norm.cdf((x - mu) / sigma))


def weighted_quantile(values,
					  quantiles,
					  sample_weight=None,
					  values_sorted=False,
					  old_style=False):
	""" Very close to np.percentile, but supports weights.
	Source: https://stackoverflow.com/questions/21844024/weighted-percentile-using-np
	NOTE: quantiles should be in [0, 1]!
	:param values: np.array with data
	:param quantiles: array-like with many quantiles needed
	:param sample_weight: array-like of the same length as `array`
	:param values_sorted: bool, if True, then will avoid sorting of initial array
	:param old_style: if True, will correct output to be consistent with np.percentile.
	:return: np.array with computed quantiles.
	"""
	values = np.array(values)
	quantiles = np.array(quantiles)
	if sample_weight is None:
		sample_weight = np.ones(len(values))
	sample_weight = np.array(sample_weight)
	assert np.all(quantiles >= 0) and np.all(
		quantiles <= 1), 'quantiles should be in [0, 1]'

	if not values_sorted:
		sorter = np.argsort(values)
		values = values[sorter]
		sample_weight = sample_weight[sorter]

	weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
	if old_style:
		# To be convenient with np.percentile
		weighted_quantiles -= weighted_quantiles[0]
		weighted_quantiles /= weighted_quantiles[-1]
	else:
		weighted_quantiles /= np.sum(sample_weight)
	return np.interp(quantiles, weighted_quantiles, values)


def weighted_std(values, weights):
	"""Return the weighted average and standard deviation.

	values, weights -- np ndarrays with the same shape.

	https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-np
	"""
	average = np.average(values, weights=weights)
	# Fast and numerically precise:
	variance = np.average((values - average)**2, weights=weights)
	return math.sqrt(variance)


class FullRank(object):
	'''
	Full-rank categorical variable encoder for use in patsy formulas.

	By default, patsy drops one category to avoid linear dependence, creating 
	k-1 dummy variables when there are k categories. Including FullRank when 
	specifying categorical variables will create k dummy variables.

	Example usage:

		y ~ C(x, FullRank)

	Code and comments from Nathaniel J. Smith (patsy dev), 
	https://stackoverflow.com/questions/46832637/one-hot-encoding-in-patsy/
	'''
	def __init__(self, reference=0):
		self.reference = reference
	
	# Called to generate a full-rank encoding
	def code_with_intercept(self, levels):
		return ContrastMatrix(
			np.eye(len(levels)),
			['[My.%s]' % (level,) for level in levels]
		)

	# Called to generate a non-full-rank encoding. But we don't care,
	# we do what we want, and return a full-rank encoding anyway.
    # Take that, patsy.
	def code_without_intercept(self, levels):
		return self.code_with_intercept(levels)


class LogitRegression(object):
	"""Patsy wrapper for logit model estimation and prediction.

	Example usage:

	# construct and estimate model using patsy formula
	# uses the cps pickle file under dataset processor
	cps["EarnedWage"] = (cps.WageIncomeLastYear > 0).astype(int)
	model = LogitRegression(
		"EarnedWage ~ C(Race)",
		cps,
		freq_weights=cps.Weight
	)

	# print model summary
	print(model)

	# return predicted probability of working for blacks
	prob_works = model.predict(
		pd.DataFrame({
			"Race": ["Black"]
		})
	)
	"""

	def __init__(self, formula=None, data=None, **kwargs):

		# convert all variables raised to a power to float64
		# this prevents mis-specification of probabilities in cases of variable overflow 
		# (if the original var was compressed to a smaller bit integer/float)
		if type(data) == pd.DataFrame:
			power_vars = list(set(re.findall(r'(?<=power\().+?(?=,)', formula)))
			for var in power_vars:
				data[var] = data[var].astype('float64')		

		if formula:
			y, X = patsy.dmatrices(formula, data, 1)
			self._y_design_info = y.design_info
			self._X_design_info = X.design_info
			self._model = GLM(y, X, family=Binomial(), **kwargs)
			self._fit = self._model.fit()
			self._betas = self._fit.params
			self._link = logit
		else:
			self._y_design_info = None
			self._X_design_info = None
			self._model = None
			self._fit = None
			self._betas = None
			self._link = logit

	def __repr__(self):
		return str(self._fit.summary()) if self._fit                           \
			else "Logistic regression"

	def predict(self, data, linear=False):

		if len(data) == 0:
			return []

		# identifies exponential variables from the design matrix (via the 'power' flag) and converts to float64
		# this prevents mis-specification of probabilities in cases of variable overflow 
		# (if the original var was compressed to a smaller bit integer/float)
		power_vars = list(set([
			re.search(r'(?<=power\().+?(?=,)', column).group() for column in \
			self._X_design_info.column_names if 'power' in column
		]))
		for var in power_vars:
			data[var] = data[var].astype('float64')						

		(X, ) = patsy.build_design_matrices([self._X_design_info], data)

		if not linear:
			return self._link.inverse(self._link(),
									  linear_transform(
										  np.asarray(X), self._betas))

		else:
			return linear_transform(np.asarray(X), self._betas)

	def draw(self, data, rand_engine):

		prediction = self.predict(data)

		return rand_engine.binomial(1, prediction)

	def to_pickle(self, filename):

		with open(filename, "wb") as f:
			pickle.dump((self._y_design_info, self._X_design_info, self._betas, self._link), f)

	@staticmethod
	def read_pickle(filename):
		y_design_info, X_design_info, betas, link = pickle.load(
			open(filename, "rb"))

		logit_regression = LogitRegression()
		logit_regression._y_design_info = y_design_info
		logit_regression._X_design_info = X_design_info
		logit_regression._betas = betas
		logit_regression._link = link

		return logit_regression

	def __add__(self, other):
		ret = copy(self)
		ret._betas = self._betas + other._betas
		return ret

	def __sub__(self, other):
		ret = copy(self)
		ret._betas = self._betas - other._betas
		return ret

	def __mul__(self, other):
		ret = copy(self)
		ret._betas = ret._betas * other
		return ret


class MultinomialRegression(object):
	"""Patsy wrapper for logit model estimation and prediction.

	Example usage:

	# construct and estimate model using patsy formula
	# uses the cps pickle file under dataset processor
	cps["EarnedWage"] = (cps.WageIncomeLastYear > 0).astype(int)
	model = LogitRegression(
		"EarnedWage ~ C(Race)",
		cps,
		freq_weights=cps.Weight
	)

	# print model summary
	print(model)

	# return predicted probability of working for blacks
	prob_works = model.predict(
		pd.DataFrame({
			"Race": ["Black"]
		})
	)
	"""

	def __init__(self, formula=None, data=None, weights=None, **kwargs):

		# convert all variables raised to a power to float64
		# this prevents mis-specification of probabilities in cases of variable overflow 
		# (if the original var was compressed to a smaller bit integer/float)
		if type(data) == pd.DataFrame:
			power_vars = list(set(re.findall(r'(?<=power\().+?(?=,)', formula)))
			for var in power_vars:
				data[var] = data[var].astype('float64')

		if formula:
			y, X = patsy.dmatrices(formula, data, 1)
			self._y_design_info = y.design_info
			self._X_design_info = X.design_info
			self._model = mnlogit(formula, data, **kwargs)
			self._fit = self._model.fit(maxiter=10000)
			self._betas = self._fit.params
		else:
			self._y_design_info = None
			self._X_design_info = None
			self._model = None
			self._fit = None
			self._betas = None

	def __repr__(self):
		return str(self._fit.summary()) if self._fit                           \
			else "Multinomial regression"

	def predict(self, data, linear=False):

		if len(data) == 0:
			return []

		# identifies exponential variables from the design matrix (via the 'power' flag) and converts to float64
		# this prevents mis-specification of probabilities in cases of variable overflow 
		# (if the original var was compressed to a smaller bit integer/float)
		power_vars = list(set([
			re.search(r'(?<=power\().+?(?=,)', column).group() for column in \
			self._X_design_info.column_names if 'power' in column
		]))
		for var in power_vars:
			data[var] = data[var].astype('float64')						

		(X, ) = patsy.build_design_matrices([self._X_design_info], data)

		# apply betas to data
		linear_transforms = linear_transform(np.asarray(X), np.asarray(self._betas))
		linear_transforms = np.concatenate(
			[np.zeros((len(data), 1)), linear_transforms], axis=1)
		linear_transforms = np.exp(linear_transforms)

		rescaled_data = pd.DataFrame(linear_transforms / np.sum(linear_transforms, axis=1, keepdims=True))

		return rescaled_data	

	def draw(self, data, rand_engine):

		prediction = self.predict(data).values.cumsum(axis=1)
		prediction = np.append(prediction, np.ones((prediction.shape[0], 1)), axis=1)
		random = rand_engine.uniform(size=(len(data), 1))
		return np.argmax(prediction > random, axis=1)

	def to_pickle(self, filename):

		pickle.dump((self._y_design_info, self._X_design_info, self._betas),
					open(filename, "wb"))

	@staticmethod
	def read_pickle(filename):
		y_design_info, X_design_info, betas = pickle.load(open(filename, "rb"))

		multinomial_regression = MultinomialRegression()
		multinomial_regression._y_design_info = y_design_info
		multinomial_regression._X_design_info = X_design_info
		multinomial_regression._betas = betas

		return multinomial_regression

	def __add__(self, other):
		ret = copy(self)
		ret._betas = self._betas + other._betas
		return ret

	def __sub__(self, other):
		ret = copy(self)
		ret._betas = self._betas - other._betas
		return ret

	def __mul__(self, other):
		ret = copy(self)
		ret._betas = ret._betas * other
		return ret


class LinearRegression(object):
	'''
	Patsy wrapper for linear estimation and prediction.

	Uses statsmodels WLS to allow weights.
	If no weights are provided, results are equivalent to OLS.
	'''

	def __init__(self, formula=None, data=None, **kwargs):

		# convert all variables raised to a power to float64
		# this prevents mis-specification of probabilities in cases of variable overflow 
		# (if the original var was compressed to a smaller bit integer/float)
		if type(data) == pd.DataFrame:
			power_vars = list(set(re.findall(r'(?<=power\().+?(?=,)', formula)))
			for var in power_vars:
				data[var] = data[var].astype('float64')

		if formula:
			y, X = patsy.dmatrices(formula, data, 1)

			self._y_design_info = y.design_info
			self._X_design_info = X.design_info

			self._model = WLS(y, X, **kwargs)
			self._fit = self._model.fit()
			self._betas = self._fit.params
			self._std = np.std(data[self._model.data.ynames].values - self.predict(data))
			self._r2 = self._fit.rsquared
			self._r2_adj = self._fit.rsquared_adj			
		else:
			self._y_design_info = None
			self._X_design_info = None
			self._model = None
			self._fit = None
			self._betas = None
			self._std = None
			self._r2 = None
			self._r2_adj = None			

	def __repr__(self):
		return str(self._fit.summary())

	def predict(self, data):
		'''
		Returns fitted values for the data provided.
		'''

		if len(data) == 0:
			return []

		# identifies exponential variables from the design matrix (via the 'power' flag) and converts to float64
		# this prevents mis-specification of probabilities in cases of variable overflow 
		# (if the original var was compressed to a smaller bit integer/float)
		power_vars = list(set([
			re.search(r'(?<=power\().+?(?=,)', column).group() for column in \
			self._X_design_info.column_names if 'power' in column
		]))
		for var in power_vars:
			data[var] = data[var].astype('float64')			

		(X, ) = patsy.build_design_matrices([self._X_design_info], data)

		return linear_transform(np.asarray(X), self._betas)

	def residuals(self, data):
		'''
		Returns residuals from fitting the model to the data provided.
		'''

		if len(data) == 0:
			return []

		return data[self._model.data.ynames].values - self.predict(data)

	def draw(self, data, rand_engine):
		'''
		Returns fitted values for the data provided plus a random draw
		from a normal distribution with the regression standard error.
		'''

		return self.predict(data) + rand_engine.normal(0, self._std, len(data))

	def Rsquared(self, adjusted=True):
		'''
		Returns the model's adjusted R squared.
		To return unadjusted R squared, pass adjusted=False.
		'''
	
		if adjusted:
			return self._r2_adj
		else:
			return self._r2

	def to_pickle(self, filename):
		'''
		Writes basic model information to a pickle file.
		'''
		
		pickle.dump((
			self._y_design_info, 
			self._X_design_info, 
			self._betas, 
			self._std,
			self._r2,
			self._r2_adj
		),
		open(filename, "wb"))

	@staticmethod
	def read_pickle(filename):
		'''
		Reads basic model information from a pickle file.

		Returns a LinearRegression object that does not include the model 
		summary or fit object but can execute all class functions.
		'''

		y_design_info, X_design_info, betas, std, r2, r2_adj = pickle.load(
			open(filename, "rb")
		)

		linear_regression = LinearRegression()
		linear_regression._y_design_info = y_design_info
		linear_regression._X_design_info = X_design_info
		linear_regression._betas = betas
		linear_regression._std = std
		linear_regression._r2 = r2
		linear_regression._r2_adj = r2_adj

		return linear_regression

	def __add__(self, other):
		ret = copy(self)
		ret._betas = self._betas + other._betas
		return ret

	def __sub__(self, other):
		ret = copy(self)
		ret._betas = self._betas - other._betas
		return ret

	def __mul__(self, other):
		ret = copy(self)
		ret._betas = ret._betas * other
		return ret

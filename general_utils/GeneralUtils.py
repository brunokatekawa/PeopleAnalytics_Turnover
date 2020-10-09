# data manipulation
import pandas as pd
import numpy as np

# plots
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# for categorical correlations
from collections import Counter
from scipy.stats import chi2_contingency
from pyitlib import discrete_random_variable as drv
from itertools import permutations

# Algorithms - Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

# metrics
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.calibration import calibration_curve


class GeneralUtils(object):

    def __init__(self):
        pass

    def test_package(self):
        print("Hello! This is a test!")

    def set_pandas_options(self):
        # disable max columns limit
        pd.set_option('display.max_columns', None)

        # sets to not crop long vlaues for rows
        pd.set_option("display.max_colwidth", None)

        # sets format to suppress scientific notation
        pd.options.display.float_format = '{:,.6f}'.format

    def get_descriptive_statistics(self, data_set):
        # central tendency: mean, median
        mean = pd.DataFrame(data_set.apply(np.mean)).T
        median = pd.DataFrame(data_set.apply(np.median)).T

        # distribution: std, min, max, range, skew, kurtosis
        std = pd.DataFrame(data_set.apply(np.std)).T
        min_value = pd.DataFrame(data_set.apply(min)).T
        max_value = pd.DataFrame(data_set.apply(max)).T
        range_value = pd.DataFrame(
            data_set.apply(lambda x: x.max() - x.min())).T
        skewness = pd.DataFrame(data_set.apply(lambda x: x.skew())).T
        kurtosis = pd.DataFrame(data_set.apply(lambda x: x.kurtosis())).T

        # concatenates
        summary_stats = pd.concat(
            [min_value, max_value, range_value, mean, median, std, skewness, kurtosis]).T.reset_index()
        summary_stats.columns = ['attributes', 'min', 'max',
                                 'range', 'mean', 'median',
                                 'std', 'skewness', 'kurtosis']

        return summary_stats

    def cramer_v(self, var_x, var_y):
        # builds contigency matrix (or confusion matrix)
        confusion_matrix_v = pd.crosstab(var_x, var_y).values

        # gets the sum of all values in the matrix
        n = confusion_matrix_v.sum()

        # gets the rows, cols
        r, k = confusion_matrix_v.shape

        # gets the chi-squared
        chi2 = chi2_contingency(confusion_matrix_v)[0]

        # makes the bias correction
        chi2corr = max(0, chi2 - (k-1) * (r-1) / (n-1))
        kcorr = k - (k-1) ** 2 / (n-1)
        rcorr = r - (r-1) ** 2 / (n-1)

        # returns cramér V
        return np.sqrt((chi2corr/n) / min(kcorr-1, rcorr-1))

    def theils_u(self, x, y):
        s_xy = drv.entropy_conditional(x, y)
        x_counter = Counter(x)
        total_occurrences = sum(x_counter.values())
        p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
        s_x = drv.entropy(p_x)

        if s_x == 0:
            return 1

        else:
            return (s_x - s_xy) / s_x

    def get_cramer_list(self, df_cat_attributes):

        # gets the cols names
        cat_cols = df_cat_attributes.columns

        # makes the permutations between cols
        pairs = list(permutations(cat_cols, 2))

        # creates auxiliar vars to be used as index
        a = 0
        b = a + 1
        c = b + 1

        # creates an aux list
        list_aux = []

        # calculate the number of turns to be looped
        turns = len(pairs) / len(cat_cols) + 1
        turns = np.arange(turns)

        # loops to build a list that stores pairs lists
        for turn in turns:
            list_aux.append([pairs[a], pairs[b], pairs[c]])
            a += len(turns) - 1
            b = a + 1
            c = b + 1

        # creates an empty list
        list_array = []

        # creates a list of arrays that store the pair including the pair (col_a, col_a)
        for element in np.arange(len(list_aux)):
            list_array.append(np.append(list_aux[element], [
                [cat_cols[element], cat_cols[element]]], axis=0))

        # creates empty list
        cramer_list = []

        for element in list_array:
            # this list will store the calculated values for a set o permutations
            values_list = []

            # makes the cramer_v calculations and store the result in the list
            for pair in element:
                values_list.append(cramer_v(
                    df_cat_attributes[pair[0]], df_cat_attributes[pair[1]]))

            # populates the cramer_list with the calculated cramer_v for each set of permutations
            cramer_list.append(values_list)

        # moves the elements inside each list in cramer_list to their respective index positions
        for values_list in cramer_list:
            elem_to_move = values_list[-1]
            values_list.insert(cramer_list.index(values_list), elem_to_move)
            values_list.pop(-1)

        return cramer_list

    def get_theils_u_list(self, df_cat_attributes):

        # gets the cols names
        cat_cols = df_cat_attributes.columns

        # makes the permutations between cols
        pairs = list(permutations(cat_cols, 2))

        # creates auxiliar vars to be used as index
        a = 0
        b = a + 1
        c = b + 1

        # creates an aux list
        list_aux = []

        # calculate the number of turns to be looped
        turns = len(pairs) / len(cat_cols) + 1
        turns = np.arange(turns)

        # loops to build a list that stores pairs lists
        for turn in turns:
            list_aux.append([pairs[a], pairs[b], pairs[c]])
            a += len(turns) - 1
            b = a + 1
            c = b + 1

        # creates an empty list
        list_array = []

        # creates a list of arrays that store the pair including the pair (col_a, col_a)
        for element in np.arange(len(list_aux)):
            list_array.append(np.append(list_aux[element], [
                [cat_cols[element], cat_cols[element]]], axis=0))

        # creates empty list
        theils_u_list = []

        for element in list_array:
            # this list will store the calculated values for a set o permutations
            values_list = []

            # makes the theils_u calculations and store the result in the list
            for pair in element:
                values_list.append(theils_u(
                    df_cat_attributes[pair[0]], df_cat_attributes[pair[1]]))

            # populates the theils_u_list with the calculated theils_u for each set of permutations
            theils_u_list.append(values_list)

        # moves the elements inside each list in theils_u_list to their respective index positions
        for values_list in theils_u_list:
            elem_to_move = values_list[-1]
            values_list.insert(theils_u_list.index(values_list), elem_to_move)
            values_list.pop(-1)

        return theils_u_list

    def get_attribute_frequency(self, attribute):

        # creates a dictionary from value counts
        count_dict_attribute = attribute.value_counts().to_dict()

        # stores values as DataFrame
        df_count_attribute = pd.DataFrame(count_dict_attribute.items(), columns=['Value', 'Count'])

        # calculates the frequency for each value
        df_count_attribute['frequency'] = df_count_attribute['Count'] / df_count_attribute['Count'].sum()

        # gets a dictionary for the frequency
        frequency_attribute = df_count_attribute.set_index('Value').to_dict()['frequency']

        return frequency_attribute

    def get_encoded_frequencies(self, data_frame):

        # creates an empty DataFrame
        encoded_freq = pd.DataFrame()

        # creates columns for frequency
        for column in data_frame.columns:
            encoded_freq[column] = data_frame[column].map(self.get_attribute_frequency(data_frame[column]))

        return encoded_freq

    def get_classifiers_performance(self, X_train, X_test, y_train, y_test, threshold, classifiers):

        # creates empty data frame
        df_performance = pd.DataFrame()

        for clf in classifiers:
            print("Training " + type(clf).__name__ + "...")
            # fits the classifier to training data
            clf.fit(X_train, y_train)

            # predict the probabilities
            clf_probs = clf.predict_proba(X_test)

            # calculates model metrics
            clf_accuracy, clf_kappa, clf_f1, clf_auc, clf_precision, clf_recall = self.calculate_model_metrics(clf,
                                                                                                               X_test, y_test,
                                                                                                               clf_probs, threshold)

            # creates a dict
            clf_dict = {
                'model': [type(clf).__name__, '---'],
                'precision': [clf_precision, np.nan],
                'recall': [clf_recall, np.nan],
                'f1-Score': [clf_f1, np.nan],
                'ROC AUC': [clf_auc, np.nan],
                'accuracy': [clf_accuracy, np.nan],
                'cohen kappa': [clf_kappa, np.nan]}

            # concatenate Data Frames
            df_performance = pd.concat([df_performance, pd.DataFrame(clf_dict)])

        # resets Data Frame index
        df_performance = df_performance.reset_index()

        # drops index
        df_performance.drop('index', axis=1, inplace=True)

        # gets only the odd numbered rows
        rows_to_drop = np.arange(1, len(classifiers)*2, 2)

        # drops unwanted rows that have no data
        df_performance.drop(rows_to_drop, inplace=True)

        # returns performance summary
        return df_performance

    def calculate_model_metrics(self, model, X_test, y_test, model_probs, threshold):
        """
            Calculates Accuracy, F1-Score, PR AUC
        """
        # keeps probabilities for the positive outcome only
        probs = pd.DataFrame(model_probs[:, 1], columns=['prob'])

        # applies the threshold
        y_pred = probs['prob'].apply(lambda x: 1 if x > threshold else 0)

        # calculates f1-score
        f1 = f1_score(y_test, y_pred)

        # calculates accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # calculates kappa score
        kappa = cohen_kappa_score(y_test, y_pred)

        # calculates AUC
        auc_score = roc_auc_score(y_test, probs)

        # calculates the precision
        precision = precision_score(y_test, y_pred)

        # calculates the recall
        recall = recall_score(y_test, y_pred)

        return accuracy, kappa, f1, auc_score, precision, recall

    def plot_pr_auc(self, y_test, model_probs, model_name):
        """
            Plots PR AUC curve
        """

        # keep probabilities for the positive outcome only
        probs = model_probs[:, 1]

        # calculate precision and recall for each threshold
        precision, recall, _ = precision_recall_curve(y_test, probs)

        # calculates the no-skill baseline
        no_skill = len(y_test[y_test == 1]) / len(y_test)

        # plots the curve
        plt.plot([0, 1], [no_skill, no_skill],
                 linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label=model_name)

        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        # title
        plt.title('PR Curve - ' + model_name, fontsize=14)

        # show the legend
        plt.legend()

        # displays the plot
        plt.show()

    def plot_pr_curves(self, X_test, y_test, classifiers):
        # define subplots
        fig, ax = plt.subplots(figsize=(15, 10))

        for clf in classifiers:
            # predict probabilities
            clf_probs = clf.predict_proba(X_test)

            # keep probabilities for the positive outcome only
            probs = clf_probs[:, 1]

            # calculate precision and recall for each threshold
            precision, recall, _ = precision_recall_curve(y_test, probs)

            # plots the curve
            plt.plot(recall, precision, marker='.', label=type(clf).__name__)

        # calculates the no-skill baseline
        no_skill = len(y_test[y_test == 1]) / len(y_test)
        plt.plot([0, 1], [no_skill, no_skill],
                 linestyle='--', label='No Skill')

        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        # title
        plt.title('PR Curves', fontsize=14)

        # adjusts subplot
        plt.tight_layout()

        # show the legend
        plt.legend()

        # displays the plot
        plt.show()

    def plot_single_confusion_matrix(self, y_test, y_pred, model, qualifier=""):
        # calculates confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # plots confusion matrix as heatmap
        ax = sns.heatmap(cm, annot=True, fmt='g', cmap='viridis',
                         square=True, annot_kws={"size": 14})

        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        ax.title.set_text(type(model).__name__ + ' ' + str(qualifier))

    def plot_multiple_confusion_matrices(self, n_rows, n_cols, X_test, y_test, classifiers, threshold):
        # define subplots
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 10))

        for clf, ax, n in zip(classifiers, ax.flatten(), list(range(n_rows*n_cols))):

            # predict the probabilities
            clf_probs = clf.predict_proba(X_test)

            # keeps probabilities for the positive outcome only
            probs = pd.DataFrame(clf_probs[:, 1], columns=['prob_default'])

            # applied the threshold
            y_pred = probs['prob_default'].apply(
                lambda x: 1 if x > threshold else 0)

            # plots confusion matrix as heatmap
            plt.subplot(n_rows, n_cols, n+1)
            self.plot_single_confusion_matrix(y_test, y_pred, clf)

        # adjusts subplot
        plt.tight_layout()

        # displays the plot
        plt.show()

    def plot_calibration_curve(self, model, y_test, model_probs, n_bins):

        # calculates the calibration curve - XGBoost Best
        frac_of_pos, mean_pred_val = calibration_curve(
            y_test, model_probs[:, 1], n_bins=n_bins, normalize=True)

        # sets plot size
        plt.figure(figsize=(8, 6))

        # plots y = x; perfect calibrated
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')

        # plots the curve
        plt.plot(mean_pred_val, frac_of_pos, 's-', label=type(model).__name__)

        # sets plot features
        plt.ylabel('Fraction of positives')
        plt.xlabel('Average Predicted Probability')
        plt.title('Calibration Curve')
        plt.legend()

    def plot_multiple_calibration_curves(self, models, y_test, models_probs, n_bins):

        # sets plot size
        plt.figure(figsize=(8, 6))

        # plots y = x; perfect calibrated
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')

        # plots calibration curve for each model
        for model, model_probs in zip(models, models_probs):

            # calculates the calibration curve
            frac_of_pos, mean_pred_val = calibration_curve(
                y_test, model_probs[:, 1], n_bins=n_bins, normalize=True)

            # plots the curve
            plt.plot(mean_pred_val, frac_of_pos,
                     's-', label=type(model).__name__)

            # sets plot features
            plt.ylabel('Fraction of positives')
            plt.xlabel('Average Predicted Probability')
            plt.title('Calibration Curve')
            plt.legend()

    def ecdf(self, data):
        """Compute ECDF for a one-dimensional array of measurements."""
        # Number of data points: n
        n = len(data)

        # x-data for the ECDF: x
        x = np.sort(data)

        # y-data for the ECDF: y
        y = np.arange(1, n+1) / n

        return x, y

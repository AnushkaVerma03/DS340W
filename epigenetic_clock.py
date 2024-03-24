'''
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import gzip


class OpenSeriesMatrix:
    """Simple class to iterate over series_matrix_files
    Arguments:
        series (str): path to series file
    Attributes:
        self.f (object): read object
        self.process_line: decodes line if necessary and returns processed list
        self.__iter__: iteration method
        """

    def __init__(self, series=None):
        # if file ends with .gz open as a binary file, else open at txt file
        if series.endswith(".gz"):
            self.f = gzip.open(series, 'rb')
        else:
            self.f = open(series, 'r')

    def __iter__(self):
        with self.f as cg:
            while True:
                line = cg.readline()
                # if line is blank break loop
                if not line:
                    break
                yield self.process_line(line)

    def process_line(self, line):
        if isinstance(line, bytes):
            return line.decode('utf-8').replace('\n', '').split('\t')
        else:
            return line.replace('\n', '').split('\t')
        
        

class SeriesMatrixParser:
    """Class to parse series matrix files into three components;
    file information, phenotype information, and methylation data.
    Arguments:
        series_matrix_path (str): path to series matrix file
        description_ids (list): list of identifiers for file descriptors
        sample_id (str): identifier for line listing sample names
        phenotype_ids(list): list of identifiers for phenotype information
        matrix_start (str): identifier list the start of the methylation matrix
    Attributes:
        self.series_matrix (OpenSeriesMatrix): iterator object
        self.series_description (dict): dict of file decriptors
        self.sample_ids (list): list of sample names
        self.phenotype_matrix {dict}: dict of phenotype information
        self.matrix_trigger (bool): bool to set beginning of
            methylation matrix in series matrix file
        self.matrix (list): list of methylation sites with values
        self.run (func): wrapper to get series matrix info
    """

    def __init__(self, series_matrix_path=None):
        assert(isinstance(series_matrix_path, str))
        self.series_matrix = OpenSeriesMatrix(series_matrix_path)
        self.series_description = {}
        self.sample_ids = []
        self.phenotype_matrix = {}
        self.matrix_trigger = False
        self.matrix = []

    def run(self, description_ids=None, sample_id=None, phenotype_ids=None, matrix_start=None):
        assert(isinstance(description_ids, list))
        assert(isinstance(sample_id, str))
        assert(isinstance(phenotype_ids, list))
        assert(isinstance(matrix_start, str))
        for line in self.series_matrix:
            if not self.matrix_trigger:
                self.get_descriptive_lines(line, description_ids)
                self.get_sample_ids(line, sample_id)
                self.get_phenotype_info(line, phenotype_ids)
                self.get_matrix(line, matrix_start=matrix_start)
            else:
                self.get_matrix(line)

    def get_descriptive_lines(self, line, description_ids):
        if line[0] in description_ids:
            try:
                info = self.series_description[line[0]]
            except KeyError:
                self.series_description[line[0]] = line[1:]
            else:
                self.series_description[line[0]] = info + line[1:]

    def get_sample_ids(self, line, sample_id):
        if line[0] == sample_id:
            self.sample_ids = line[1:]

    def get_phenotype_info(self, line, phenotype_ids):
        if line[0] in phenotype_ids:
            phenotype_label = line[1].split(':')[0].strip(' "')
            self.phenotype_matrix[phenotype_label] = []
            for phenotype in line[1:]:
                phenotype_split = phenotype.split(':')
                self.phenotype_matrix[phenotype_label].append(phenotype_split[1].strip(' "'))

    def get_matrix(self, line, matrix_start=None):
        if self.matrix_trigger:
            self.matrix.append(line)
        elif line[0] == matrix_start:
            self.matrix_trigger = True
            
            
# name geo file
geo_file = 'GSE41169_series_matrix.txt'

# run parser class on the downloaded information, you will have to
# identify phenotype information and descriptors manually
example_matrix = SeriesMatrixParser(f'{geo_file}')
example_matrix.run(description_ids=['!Series_title', '!Series_geo_accession',
                                    '!Series_pubmed_id', '!Series_summary',
                                    '!Series_overall_design',
                                    '!Series_sample_id', '!Series_relation'],
               sample_id='!Sample_geo_accession',
               phenotype_ids=['!Sample_characteristics_ch1'],
               matrix_start='!series_matrix_table_begin')

# transform matrix list into a pandas dataframe
example_matrix_df = pd.DataFrame(data=example_matrix.matrix[1:-1],
                                columns=example_matrix.matrix[0])

# set index
example_matrix_df = example_matrix_df.set_index('"ID_REF"')

# transform strings to float values
example_matrix_df = example_matrix_df.apply(pd.to_numeric, errors='coerce')

# drop rows, methylation sites, with missing infromation
example_matrix_df = example_matrix_df.dropna(axis=0)


# retrieve age phenotype
example_matrix_age = [int(x) for x in example_matrix.phenotype_matrix['age']]

# define a PCA object
qc_pca = PCA(n_components=4, whiten=False)

# fit the PCA object
qc_pca_values = qc_pca.fit_transform(example_matrix_df.values.T)

# get the variance explained for the first two principal components
variance_explained = qc_pca.explained_variance_ratio_

pc1 = qc_pca_values[:,0]
pc2 = qc_pca_values[:,1]

# scatter plot of first two PCs, and retrieve and sample outliers
# list to store sample outliers
non_outlier_list = []
non_outlier_age = []
fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(pc1, pc2,)
ax.set_title('Methylation Matrix PCA')
ax.set_xlabel(f'PC1 Variance Explained = {variance_explained[0]:0.3f}')
ax.set_ylabel(f'PC2 Variance Explained = {variance_explained[1]:0.3f}')
# iterate through pc1, pc2, and sample labels to add labels to plotted points
for x, y, label, age in zip(pc1, pc2, list(example_matrix_df), example_matrix_age):
    ax.text(x=x, y=y, s=label)
    # add outliers to list
    if x < 5:
        non_outlier_list.append(label)
        non_outlier_age.append(age)
plt.show()


# want a list of sample names
sample_ids = list(example_matrix_df[non_outlier_list])

X_train, X_test, y_train, y_test = train_test_split(sample_ids, non_outlier_age, test_size=0.1)

# take dataframe values as a numpy array and transpose the array with .T
X_train = example_matrix_df[X_train].values.T
X_test = example_matrix_df[X_test].values.T


# initialize a penalized regression object
lasso_cv = LassoCV(cv=3, n_jobs=2)

# fit object
lasso_cv.fit(X_train, y_train)

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

predicted_test_age = lasso_cv.predict(X_test)
test_score = r2(predicted_test_age, y_test)
'''







import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('patientlungcancer.csv')

X = df.drop('Age', axis=1)
y = df['Age'] 

categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


model = RandomForestRegressor(n_estimators=100, random_state=0)

# pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_test)

score_mae = mean_absolute_error(y_test, preds)
score_mse = mean_squared_error(y_test, preds)
score_r2 = r2_score(y_test, preds)

print('MAE:', score_mae)
print('MSE:', score_mse)
print('R^2:', score_r2)

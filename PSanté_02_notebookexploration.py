#!/usr/bin/env python
# coding: utf-8

# # P3 : Concevez une application au service de la santé publique

# ## Notebook Exploration

# Pascaline Grondein
# 
# Début de projet : 07/03/2022
# 
# 
# L'agence "Santé publique France" a lancé un appel à projets pour trouver des idées innovantes d’applications en lien avec l'alimentation, à l’aide du jeu de données Open Foods.
# 
# 
# L'application proposée ici est appelée **Made In FooD**. Elle propose de s'adapter au régime alimentaire de son utilisateur.rice, de noter les produits en fonction de ce régime ainsi que de son enpreinte carbone, avec un bonus si le produit est Made In France. L'application peut également proposer une alternative, si elle existe, pour un produit moins salé, moins sucré, avec une empreinte carbone plus faible...
# 
# 
# Ce notebook explore le jeu de données, analyse les variables pertinentes et en tire les conclusions nécessaires sur la faisabilité de l'application Made In FooD. 

# ### Table of Contents
# 
# * [I. Récupération des données nettoyées](#chapter1)
# * [II. Analyse univariée](#chapter2)
#     * [1. Variable qualitative : nutrition grade](#section_2_1)
#     * [2. Variables quantitatives discrètes](#section_2_2)
#         * [a. Nutrition score](#section_2_2_1)
#         * [b. Additifs](#section_2_2_2)
#         * [c. Huile de palme](#section_2_2_3)
#     * [3. Variables quantitatives continues : Macro & nutriments](#section_2_3)
# * [III. Analyse bivariée et multivariée](#chapter3)
#     * [1. Corrélation variable qualitative/quantitative](#section_3_1)
#     * [2. Corrélation variable quantitative/quantitative](#section_3_2)
#         * [a. Pairplot](#section_3_2_1)
#         * [b. Heatmap](#section_3_2_2)
#         * [c. Matrice de corrélation et tests statistiques](#section_3_2_3)
#         * [d. Régression linéaire](#section_3_2_4)
#     * [3. ACP](#section_3_3)
# * [IV. Conclusion](#chapter4)

# In[102]:


#importation des fonctions utiles
from nettoyage import *
from ACP import *


#importation librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import decomposition
from sklearn import preprocessing
import statsmodels.api as sm

from scipy.stats import pearsonr
from statsmodels.formula.api import ols

import warnings
warnings.filterwarnings("ignore")


# # <a class="anchor" id="chapter1">I. Récupération des données nettoyées</a>

# In[103]:


#Importation du fichier tout en précisant le type de séparation
data = pd.read_csv('fr_openfoodfacts_org_products.csv', sep='\t',decimal='.',low_memory=False)
data_subset = data[['code','url','product_name','generic_name','brands','brands_tags','categories','categories_tags','countries_fr','main_category_fr','nutrition-score-fr_100g','proteins_100g','carbohydrates_100g','sugars_100g','fat_100g','saturated-fat_100g','sodium_100g','origins','origins_tags','labels','labels_tags','manufacturing_places','manufacturing_places_tags','packaging','first_packaging_code_geo','ingredients_from_palm_oil','ingredients_from_palm_oil_n','ingredients_that_may_be_from_palm_oil','ingredients_that_may_be_from_palm_oil_n','additives_n','carbon-footprint_100g','emb_codes','emb_codes_tags','traces','nutrition_grade_fr']]
data_subset.astype({'ingredients_from_palm_oil': 'object','ingredients_that_may_be_from_palm_oil':'object',}).dtypes


# In[104]:


#nettoyage du jeu de données
data_subset = nettoyage(data_subset,pct_seuil_col = 50, pct_seuil_row = 50, itimp = 0,lower_percentile = 0.05,higher_percentile = 0.95)


# #  <a class="anchor" id="chapter2">II. Analyse univariée</a>

# In[105]:


data_subset.dtypes


# Les variables code, url, product name, brands sont des variables d'identification.
# 
# Les variables utiles pour l'app sont le nutrition_grade_fr, nutrition-score-fr_100g, proteins_100g, carbohydrates_100g, sugars_100g, fat_100g, saturated-fat_100g, sodium_100g, ingredients_from_palm_oil_n, ingredients_that_may_be_from_palm_oil_n, additives_n.

# In[106]:


df = data_subset


# ##  <a class="anchor" id="section_2_1">1. Variable qualitative : Nutrition grade</a>

# In[107]:


df['nutrition_grade_fr'].describe()


# In[108]:


df['nutrition_grade_fr'].value_counts()


# In[109]:


plot = df['nutrition_grade_fr'].value_counts(normalize=True).plot(kind='pie',y='nutrition_grade_fr',labels = df['nutrition_grade_fr'].unique(), autopct='%1.0f%%',figsize=(10, 10),fontsize = 20, legend = True )
plot.set_title('Répartition du nutrigrade',fontsize = 30)
plt.legend(fontsize = 15)
plt.axis('equal') 
plt.show() 


# On observe avec ce pie plot que plus de 40% des produits ont un nutrigrade de d. Les autres grades sont répartis de façon plus ou moins égale, autour de 15%.

# ## <a class="anchor" id="section_2_2">2. Variables quantitatives discrètes </a>

# ###  <a class="anchor" id="section_2_2_1">a. Nutrition score</a>

# In[110]:


df['nutrition-score-fr_100g'].describe()


# In[111]:


sns.set_style("whitegrid")
plt.figure(figsize=(10,8))

boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1, color='k')
meanprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick')

df.boxplot(column="nutrition-score-fr_100g",boxprops = boxprops,showfliers=True, medianprops=medianprops,vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
plt.title("Nutriscore",fontsize = 30)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Score',fontsize = 25)
plt.ylabel('',fontsize = 25)

plt.show()


# In[112]:


df_nutri = pd.DataFrame(data = df['nutrition-score-fr_100g'].value_counts(normalize = True)).sort_index()
plot = df_nutri.plot.bar(y ='nutrition-score-fr_100g',rot=90,legend = False,figsize = (20,10),)

figsize = (25,20)
plt.title("Nutriscore",fontsize = 40)
plt.xticks(fontsize = 15)
#plot.set_xlim(-15.0, 40.0)
plt.yticks(fontsize = 15)
plt.xlabel('Score',fontsize = 25)
plt.ylabel('%',fontsize = 25)

plt.show()


# La distribution du nutriscore semble indiquer une importante proportion de produits possèdant la note 10, cependant, il est possible que ce soit la conséquence du traitement des valeurs manquantes par la médiane. 

# ### <a class="anchor" id="section_2_2_2">b. Additifs</a>

# In[113]:


df['additives_n'].value_counts()


# In[114]:


df['additives_n'].describe()


# In[115]:


sns.set_style("whitegrid")
plt.figure(figsize=(10,8))

boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1, color='k')
meanprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick')

df.boxplot(column="additives_n",boxprops = boxprops,showfliers=True, medianprops=medianprops,vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
plt.title("Additifs",fontsize = 30)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('Nombre',fontsize = 25)
plt.ylabel('',fontsize = 25)

plt.show()


# In[116]:


df_add = pd.DataFrame(data = df['additives_n'].value_counts(normalize = True)).sort_index()
plot = df_add.plot.bar(y ='additives_n',rot=0,legend = False,figsize = (20,10))

figsize = (25,20)
plt.title("Additifs",fontsize = 40)
plt.xticks(fontsize = 25)
#plot.set_xlim(-15.0, 40.0)
plt.yticks(fontsize = 25)
plt.xlabel('Nombre',fontsize = 25)
plt.ylabel('%',fontsize = 25)

plt.show()


# Ce graphe semble indiquer que près de 40% des produits n'ont pas d'additifs parmi leurs composants, cependant, comme pour le nutriscore, il est possible que le traitement des valeurs manquantes ait eu un impact. Par souscis de précision, je ne sélectionnerais pas cette variable pour l'application.

# ### <a class="anchor" id="section_2_2_3">c. Huile de palme</a>

# In[117]:


df['ingredients_from_palm_oil_n'].value_counts()


# In[118]:


df['ingredients_that_may_be_from_palm_oil_n'].value_counts()


# Les variables liées à l'huile de palme sont trop peu renseignées pour être analysées et utilisées pour l'application. 

# ##  <a class="anchor" id="section_2_3">3. Variables quantitatives continues : Macros & nutriments </a>

# In[119]:


macro = ['proteins_100g','carbohydrates_100g','sugars_100g','fat_100g','saturated-fat_100g','sodium_100g']


# In[120]:


df_macro = df[['proteins_100g','carbohydrates_100g','sugars_100g','fat_100g','saturated-fat_100g','sodium_100g']]
df_macro.describe()


# In[121]:


sns.set_style("whitegrid")
plt.figure(figsize=(10,8))

boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1, color='k')
meanprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick')

meds_norm = df_macro.median()
meds_norm.sort_values(ascending=False, inplace=True)
df_macro = df_macro[meds_norm.index]

df_macro.boxplot(boxprops = boxprops,showfliers=True, medianprops=medianprops,vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
plt.title("Macro & nutriments",fontsize = 30)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('',fontsize = 25)
plt.ylabel('',fontsize = 25)

plt.show()


# In[122]:


for var in macro :
    
    print('Analyse univariée pour la variable {}'.format(var))
    
    #df[var].hist(density=True,bins=100,figsize = (10,8))
    df[var].where(df[var]>0).hist(density=True,bins=100,figsize = (10,8))

    plt.title('Distribution de la variable : '+var,fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Grammes',fontsize = 25)
    plt.ylabel('%',fontsize = 25)

    plt.show()


# Pour chaque variables, on remarque que la majorité des produits se situent entre 0 et 20g, exception faite des carbohydrates qui semblent possèder une distribution des valeurs plus étalée.

# # <a class="anchor" id="chapter3">III. Analyse bivariée et multivariée</a> 

# ## <a class="anchor" id="section_3_1">1. Corrélation variable qualitative/quantitative : ANOVA</a>

# In[123]:


list_quant = ['nutrition-score-fr_100g','fat_100g']


# In[124]:


#Fonction pour calculer le rapport de corrélation
def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


# In[125]:


for var in list_quant:
    
    print('---Analyse bivariée pour les variable nutrigrade et {}---'.format(var))
    
    df_nutri = df[['nutrition_grade_fr', var]]
    grouped = df_nutri.groupby(['nutrition_grade_fr'])
    df2 = pd.DataFrame({col:vals[var] for col,vals in grouped})
    meds = df2.median()
    meds.sort_values(ascending=False, inplace=True)
    df2 = df2[meds.index]

    boxprops = dict(linestyle='-', linewidth=1, color='k')
    medianprops = dict(linestyle='-', linewidth=1, color='k')
    meanprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick')
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10,8))
    df2.boxplot(labels=df_nutri['nutrition_grade_fr'].unique(), boxprops = boxprops,showfliers=True, medianprops=medianprops,vert=True, patch_artist=True, showmeans=True, meanprops=meanprops)
    plt.title("{} en fonction du nutrigrade".format(var),fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show()
    
    print('   ')
    print('Le rapport de corrélation est de',eta_squared(df['nutrition_grade_fr'],df[var]))
    print('   ')


# On observe que le nutriscore et le nutrigrade sont très corrélés. Plus le nutriscore est haut plus la lettre est basse (e), et inversement.Il est donc possible d'utiliser le nutriscore ou le nutrigrade, mais les deux seraient redondant. Le gras et le nutrigrade sont légèrement corrélés, on observe que les produits possèdant un grade a ou b ont en moyenne moins de gras dans leur composition. 

# Effectuons un test de significativité. Posons les hypothèses:
# 
#  - H0 : Les moyennes de chaque groupe sont égales si p-value > 5%
#  - H1 : Les moyennes de chaque groupe ne sont pas toutes égales si p-value < 5%
# 

# In[126]:


df_anova = df[['nutrition_grade_fr','nutrition-score-fr_100g','fat_100g']]
df_anova.rename(columns = {'nutrition_grade_fr':'nutrigrade', 'nutrition-score-fr_100g':'nutriscore'}, inplace = True)
model = ols('nutriscore ~ nutrigrade',data = df_anova).fit()
anova_table = sm.stats.anova_lm(model,typ=2)
display(anova_table)
print(model.summary())


# In[127]:


model = ols('fat_100g ~ nutrigrade',data = df_anova).fit()
anova_table = sm.stats.anova_lm(model,typ=2)
display(anova_table)
print(model.summary())


# ## <a class="anchor" id="section_3_2">2. Corrélation variable quantitatives/quantitative</a>

# In[128]:


df_quant = df[['additives_n','nutrition-score-fr_100g','proteins_100g','carbohydrates_100g','sugars_100g','fat_100g','saturated-fat_100g','sodium_100g']]
#df_quant


# ### <a class="anchor" id="section_3_2_1">a. Pairplot</a>

# In[129]:


sns.pairplot(df_quant,plot_kws=dict(marker=".", linewidth=1),diag_kind = 'hist',height = 2)
#plt.xticks(fontsize = 15)
#plt.yticks(fontsize = 15)
plt.show()


# ### <a class="anchor" id="section_3_2_2">b. Heatmap</a> 

# In[130]:


plt.figure(figsize=(10,8))
sns.heatmap(df_quant.corr(), annot = True, fmt='.2g',cmap= 'YlGnBu')
plt.show()


# Grace à la matrice de corrélation, on voit plus clairement que les couples sucres/carbohydrates, saturated fat/fat, nutriscore/saturated_fat et sugars_100g/nutriscore sont corrélés à un certain degré.

# ### <a class="anchor" id="section_3_2_3">c. Matrice de corrélation et tests statistiques</a>  

# In[131]:


df_corr = df_quant.corr(method='pearson')
df_corr


# Effectuons des tests statistiques sur ces variables. Posons les hypothèses :
# 
#  - H0 : Variables indépendantes si p-value > 5%
#  - H1 : Variables non indépendantes si p-value < 5%
# 
# Calculons maintenant la matrice des p-values.

# In[137]:


a = np.empty((len(df_quant.columns),len(df_quant.columns),))
a[:] = np.nan
for i in range(0,len(df_quant.columns)):
    for j in range(0,len(df_quant.columns)):
        a[i,j] = pearsonr(df_quant.iloc[:,i], df_quant.iloc[:,j])[1]

df_pvalue = round(pd.DataFrame(a, columns=df_quant.columns, index = df_quant.columns),5)


# In[140]:


cm = sns.light_palette("green", as_cmap=True) 

df_pvalue.style.background_gradient(cmap=cm).set_precision(5)


# ### <a class="anchor" id="section_3_2_4">d. Régression linéaire</a> 

# On observe les couple de variables qui semblaient corrélées en étudiant la heatmap.

# In[145]:


seuil_corr = 0.4
corr_ = []
for col in df_corr:
    for row in df_corr:
        corr = df_corr.loc[row,col]
        #corr_.append(corr)
        if corr not in corr_:
            
            if (abs(corr) > seuil_corr) & (corr < 1):
            
                corr_.append(corr)
                
                print('----Analyse bivariée pour les variable {}'.format(row),'et {}.----'.format(col))
            
                Y = df_quant[col]
                X = df_quant[[row]]

                X = X.copy() # On modifiera X, on en crée donc une copie
                X['intercept'] = 1.
                result = sm.OLS(Y, X).fit() # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
                a,b = result.params[row],result.params['intercept']
            
                print('  ')
                print('La valeur de R² est ',result.rsquared)
                print('La p-value est de',result.pvalues[1])
                print('  ')
                
            
                sns.set_style("whitegrid")
                plt.figure(figsize=(10,8))
                plt.plot(df_quant[[row]],df_quant[col], ".")
                plt.plot(np.arange(100),[a*x+b for x in np.arange(100)],color = 'red')
                plt.xlabel(row, fontsize = 20)
                plt.ylabel(col, fontsize = 20)
                plt.xticks(fontsize = 15)
                plt.yticks(fontsize = 15)
                plt.show()
                print('a = ',a,', b = ',b)
                print('  ')
                print('***************')
                print('  ')
            


# Les variables saturated_fat/fat sugars/carbohydrates semblent assez corrélées. Il serait alors possible de ne considérer que carbohydrates et fat comme variables principales pour ces couples.

# ## <a class="anchor" id="section_3_3">3. ACP</a> 

# In[51]:


# choix du nombre de composantes à calculer
n_comp = 6


# In[52]:


names = df_quant.index 
features = df_quant.columns
X = df_quant.values
#X


# In[53]:


#Centrage et réduction
X_scaled = preprocessing.StandardScaler().fit_transform(X) 


# In[54]:


# Calcul des composantes principales
pca = decomposition.PCA(n_components=n_comp) #calcul des composantes PCA
pca.fit(X_scaled) #Fit the model with X_scaled


# In[56]:


#Calcul du seuil d'inertie
#inertie_seuil = (100/len(df_quant.columns))
#print('Considérons les axes dont l\'inertie associée est supérieure à',inertie_seuil,'%.')


# In[57]:


# Eboulis des valeurs propres
#plt.figure(figsize=(10,8))
display_scree_plot(pca)


# Par ce graphe on constate que les deux premières composantes couvrent 50% de l'inertie de la population. En utilisant également les composantes 3 et 4, on atteint quasiment 80%. 

# In[61]:


#Cercle des corrélations
pcs = pca.components_ #Principal axes in feature space, representing the directions of maximum variance in the data.  The components are sorted by explained_variance_.


# In[62]:


display_circles(pcs, n_comp, pca, [(0,1),(2,3)], labels = np.array(features))


# In[63]:


# Projection des individus
X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)], labels = None)

plt.show()


# # <a class="anchor" id="chapter4">IV. Conclusion</a> 

# Pour l'application, puisque les variables rendant compte de l'empreinte carbone, de l'huile de palme et du nombre d'additifs ne sont pas assez renseignée. On laisse de côté l'idée de Made in Food.
# 
# On peut cependant se concentrer sur une classification des produits en fonction de 4 variables clés. Pour les deux premières, je prendrais carbohydrates_100g et fat_100g, la première étant fortement corrélées à saturated_fat_100g et la seconde à sugars_100g. Les deux autres variables que je retiendrais seraient proteins_100g et sodium_100g, l'apport en protéine étant une donnée intéressante d'un point de vue nutritionnel, et le sel étant un élément pouvant demander une monitorisation constante, de nombreux régime sans sel étant préconisé pour diverses raisons médicales (traitement, maladie,...). En bonus, il serait intéressant d'afficher le nutrigrade associé au produit, comme information complémentaire. 
# 
# On peut alors imaginer une application demandant à son utilisateur le type de régime suivi (sans sel, hyper protéiné, faible en gras...) et qui noterait les produits en fonction de ces objectifs, tout en recommandant d'autre produit avec moins de se, moins de gras, plus de protéine...Une application répondant aux vrais besoins de ses utilisateurs.

# In[ ]:





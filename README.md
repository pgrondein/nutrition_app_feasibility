# Nutrition Application - Feasibility Report

![made_in_food](https://github.com/pgrondein/nutrition_app_feasibility/assets/113172845/a7e868b3-669f-402f-9a0a-9777e41c7217)


## Context

Proposal for a food related application. 
Exploration report for feasibility of a nutritional assessment application.

## Data
The dataset  is that of Open Food Facts, available on [the official website](https://world.openfoodfacts.org/). Features are defined at [this address](https://world.openfoodfacts.org/data/data-fields.txt).

The fields are separated into four sections:

- General information on the product sheet: name, date of modification, etc.
- A set of tags: product category, location, origin, etc.
- The ingredients making up the products and their possible additives.
- Nutritional information: quantity in grams of a nutrient per 100 grams of the product.

## Application : Made in Food
The App would work  as follow :
- The user specifies his diet if one (vegetarian, vegan, high-protein, salt-free, etc.): each feature will have a different weight on final rating depending on the selected diet, positive or negative.
- The application gives an overall score to the products according to the composition, and the user diet.
- The application can recommend a product of the same type with a better rating.

## Features
First, each product in the list must be differentiable, the identification features are therefore kept (code, product name, etc.).

In order to give an overall nutritional score to the products, it is necessary to have access to the nutritional values (proteins, carbohydrates, sugars, salt, etc.) for 100 grams of product, as well as to the nutriscore and nutrigrade provided.

In total, after data cleaning and processing, 15 variables are pre-selected. The dataset is ready for analysis, and final selection of relevant features for the project.

## Analysis
### Univariate
#### Nutrigrade is a qualitative feature.

<img src="https://github.com/pgrondein/nutrition_app_feasibility/assets/113172845/15e1ee5d-2fbc-4814-85ad-310ab2ceb678" height="400">

We observe with this pie plot that more than 40% of the products have a Nutrigrade of d. The other grades are distributed more or less evenly, around 15%.

#### Nutriscore is a discrete quantitative feature

<img src="https://github.com/pgrondein/nutrition_app_feasibility/assets/113172845/d360ab1c-6783-4845-9d46-10f0586256d6" height="400">

The distribution of the Nutriscore seems to indicate a large proportion of products with a score of 10, however, it is possible that this is the consequence of the treatment of missing values by median values.

#### Macros & nutrients are continuous quantitative features

<img src="https://github.com/pgrondein/nutrition_app_feasibility/assets/113172845/8890e1b1-0b07-49f4-a241-d2a6ad417c0f" height="400">

### Multivariate
In order to check feature redundancy, it is interesting to study the correlation level between them. Indeed, keeping two correlated features brings no added value to the final grade.
#### ANOVA (ANalysis Of VAriance) - Nutriscore & Nutrigrade
An ANOVA test is used to determine relationship degree between two features.

<img src="https://github.com/pgrondein/nutrition_app_feasibility/assets/113172845/be11fa18-ee9c-4d35-8e99-546312708663" height="400">

Mean values of each Nutriscore group appear clearly different. The higher the Nutriscore, the lower the letter (e), and inversely.

The value of eta² is 0.74, closer to 1 than to 0, which seems to indicate again that the values of each Nutriscore group are very close, and that there is a relationship between Nutriscore and Nutrigrade.

In order to confirm this idea, let us carry out a test of significance. Let's make the assumptions:

- H0: The means of each group are equal if p-value > 5%
- H1: The means of each group are not all equal if p-value < 5%

It appears that p-value < 5%, the correlation hypothesis is therefore validated. It is therefore possible to use Nutriscore or Nutrigrade, but both would be redundant.

#### Correlation between nutritional values
In order to observe the possible relationships between nutritional features, I use a heatmap which gives me an overview of all of them.

<img src="https://github.com/pgrondein/nutrition_app_feasibility/assets/113172845/051d1cbe-e825-4544-9ad0-647152576d2e" height="400">

Thanks to the correlation matrix, we see more clearly that the pairs 

- Sugars/Carbohydrates,
- Saturated Fat/Fat,
- Nutriscore/Saturated Fat
- Nutriscore/Fat
- Sugars/Nutriscore

are correlated to some degree.

For each pair, I perform a significance test.

Let's make the assumptions:

- H0: Independent variables if p-value > 5%
- H1: Non-independent variables if p-value < 5%

The calculation of the p-value gives for each couple:

- Sugars/Carbohydrates: p-value < 5%
- Saturated Fat/Fat: p-value < 5%
- Nutriscore/Saturated Fat: p-value < 5%
- Nutriscore/Fat: p-value < 5%
- Sugars/nutriscore: p-value < 5%

The Sugars, Fats and Saturated Fat features are therefore correlated with the Nutriscore. In addition, Fat and Saturated Fat are also correlated, as are Sugars and Carbohydrates. I therefore exclude Sugars and Saturated Fat from the variables for the overall score.

## Conclusion
The nutritional features selected for the overall product score are:

- Nutriscore
- Carbohydrates per 100g of product
- Fat per 100 g of product
- Protein per 100 g of product
- Salt per 100 g of product

As a bonus, it would be interesting to display the Nutrigrade associated with the product, as additional information, although the correlation with the Nutriscore is proven.

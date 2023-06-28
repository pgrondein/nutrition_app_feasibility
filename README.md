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

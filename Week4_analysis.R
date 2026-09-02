# =============================================================
# Week 4 Task: Comprehensive Data Analysis Reporting
# Dataset: Titanic Passenger Data (891 records)
# Integrates: Week 1 (cleaning) + Week 2 (visualization) + Week 3 (modeling)
# =============================================================

library(corrplot)
library(caret)
library(pROC)
library(ggplot2)

# ---------------------------------------------------------------
# PART A. DATA PREPARATION (recap of Week 1 pipeline)
# ---------------------------------------------------------------
titanic <- read.csv("titanic.csv", stringsAsFactors = FALSE)

titanic$Has_Cabin <- ifelse(titanic$Cabin == "" | is.na(titanic$Cabin), 0, 1)
titanic$Cabin <- NULL

age_median <- median(titanic$Age, na.rm = TRUE)
titanic$Age[is.na(titanic$Age)] <- age_median

embarked_mode <- names(sort(table(titanic$Embarked[titanic$Embarked != ""]),
                             decreasing = TRUE))[1]
titanic$Embarked[titanic$Embarked == "" | is.na(titanic$Embarked)] <- embarked_mode

fare_cap <- quantile(titanic$Fare, 0.99, na.rm = TRUE)
titanic$Fare_capped <- ifelse(titanic$Fare > fare_cap, fare_cap, titanic$Fare)

titanic$Sex_encoded <- ifelse(titanic$Sex == "male", 1, 0)
titanic$Pclass <- factor(titanic$Pclass, levels = c(1, 2, 3), ordered = TRUE)
titanic$FamilySize <- titanic$SibSp + titanic$Parch + 1
titanic$IsAlone <- ifelse(titanic$FamilySize == 1, 1, 0)

# ---------------------------------------------------------------
# PART B. KEY VISUALIZATIONS (recap of Week 2 outputs)
# ---------------------------------------------------------------
ggplot(titanic, aes(x = Sex, fill = factor(Survived))) +
  geom_bar(position = "fill") +
  labs(title = "Survival Rate by Sex", y = "Proportion", fill = "Survived")

ggplot(titanic, aes(x = factor(Pclass), fill = factor(Survived))) +
  geom_bar(position = "fill") +
  labs(title = "Survival Rate by Passenger Class", y = "Proportion", fill = "Survived")

num_vars <- titanic[, c("Survived", "Age", "Fare_capped", "SibSp", "Parch",
                         "FamilySize", "Sex_encoded")]
corrplot(cor(num_vars), method = "color", addCoef.col = "black")

# ---------------------------------------------------------------
# PART C. STATISTICAL HYPOTHESIS TESTING (Week 3)
# ---------------------------------------------------------------
chisq.test(table(titanic$Sex, titanic$Survived))
chisq.test(table(titanic$Pclass, titanic$Survived))
t.test(Age ~ Survived, data = titanic)
shapiro.test(sample(titanic$Fare_capped, 500))

# ---------------------------------------------------------------
# PART D. PREDICTIVE MODEL: LOGISTIC REGRESSION (Week 3)
# ---------------------------------------------------------------
set.seed(42)
train_idx <- createDataPartition(titanic$Survived, p = 0.8, list = FALSE)
train <- titanic[train_idx, ]
test  <- titanic[-train_idx, ]

model <- glm(Survived ~ Pclass + Sex_encoded + Age + Fare_capped +
               SibSp + Parch + FamilySize,
             data = train, family = binomial)
summary(model)

pred_prob <- predict(model, newdata = test, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

confusionMatrix(factor(pred_class), factor(test$Survived), positive = "1")
roc_obj <- roc(test$Survived, pred_prob)
auc(roc_obj)
plot(roc_obj, main = "ROC Curve - Logistic Regression")

# ---------------------------------------------------------------
# PART E. MODEL DIAGNOSTICS
# ---------------------------------------------------------------
par(mfrow = c(2, 2))
plot(model)   # residuals vs fitted, Q-Q, scale-location, leverage
par(mfrow = c(1, 1))

exp(coef(model))   # odds ratios

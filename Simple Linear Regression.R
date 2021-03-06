#Simple Linear Regression

dataset = read.csv('Salary_Data.csv')
#dataset = dataset[,2:3]

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split ==  FALSE)

# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

y_pred = predict(regressor, newdata = test_set)

ggplot() +
  geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary),
             colour='red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = train_set)),
            colour='blue') +
  ggtitle('Salary vs Experience(Training set)')+
  xlab('Years of experience') +
  ylab('Salary')

ggplot() +
  geom_point(aes(x=test_set$YearsExperience, y=test_set$Salary),
             colour='red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata = train_set)),
            colour='blue') +
  ggtitle('Salary vs Experience(Test set)')+
  xlab('Years of experience') +
  ylab('Salary')



  

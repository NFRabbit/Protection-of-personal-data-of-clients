# Protection-of-personal-data-of-clients

we need to protect the data of clients for the insurance company "Hot Pot." Develop a method for data transformation that makes it difficult to recover personal information. Additionally, explain the correctness of the method's operation. It is necessary to ensure that the quality of machine learning models does not decrease during data transformation, and finding the best model is not required.

 - Пол - Gender

 - Возраст- Age

 - Зарплата - Salary

 - Члены семьи - Family members

 - Страховые выплаты - Insurance payments

Conclusion:
In the case of multiplying the feature matrix by a random invertible matrix, it is possible to protect the data while not losing much in the quality of the model. (The slight difference is caused by the peculiarity of the matrix transformations and floating-point arithmetic, which is normal.)

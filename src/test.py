from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
print(le.fit(["paris", "paris", "tokyo", "amsterdam"]))

list(le.classes_)
print(list(le.classes_))

le.transform(["tokyo", "tokyo", "paris"])
print(le.transform(["tokyo", "tokyo", "paris"]))

list(le.inverse_transform([2, 2, 1]))
print(list(le.inverse_transform([2, 2, 1])))

print(le.fit_transform(["tokyo", "tokyo", "paris", "amsterdam", 'ggg']))
print(le.transform(['ggg']))

b = ['SFASFAF', 'A']
print(b[0].lower())

a=[1,2,3]
b=a
b[0]=2
print(a)
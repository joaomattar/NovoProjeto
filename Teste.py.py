import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 30)
pd.set_option('display.min_rows', 15)
from sklearn.preprocessing import scale
from scipy import stats as st
import pingouin as pg


#Importando os dados de seu respectivos anos

dados = pd.read_csv("E:\\Estudos\\Data Science\\Fase 05\\vendas_linha_petshop_2019.csv", decimal=',', sep=';')

print(dados.head())
print(dados.dtypes)

dados['quantidade2'] = dados['quantidade'].fillna(dados['valor_total_bruto'] / dados['valor'])

print(dados['quantidade2'])

dados['quantidade2'] = dados['quantidade2'].astype(int)

dados['score-z'] = scale(dados['quantidade2'])

print(dados.sort_values('score-z', ascending=False))

#Desvio médio absoluto

print(dados.groupby('produto') \
    .agg(desvio_medio = pd.NamedAgg('quantidade2', 'std')))

print(dados.groupby('produto') \
    .agg(variancia = pd.NamedAgg('quantidade2', 'var'),
         desvio_padrao = pd.NamedAgg('quantidade2', 'std')).reset_index())

print(dados['quantidade2'].quantile([0.9]))

win = dados['quantidade2'].quantile([0.9]).to_list()

print(dados.sort_values('score-z', ascending=False))

lista = []

for i in dados['quantidade2']:
    lista.append(i)

print(lista)
lista2 = []

for i in lista:
    if i > 4:
        i = 3
        lista2.append(i)
    else:
        lista2.append(i)

print(lista2)

teste = pd.DataFrame(lista2,
                     columns=['quantidade6']).reset_index()

print(teste)

#teste.boxplot(column='quantidade6')
#plt.show()

teste['score-z'] = scale(teste['quantidade6'])

print(teste.sort_values('score-z', ascending=False))

print(teste.describe())

print(teste['quantidade6'].var())


print(dados[dados['score-z'] > 3])


print(dados['centro_distribuicao'].unique())


novo = dados.groupby('centro_distribuicao') \
    .agg(media = pd.NamedAgg('quantidade2', 'mean'),
         desvio_padrao = pd.NamedAgg('quantidade2', 'std'),
         n = pd.NamedAgg('quantidade2', 'count')).reset_index()

e1 = dados[dados['centro_distribuicao'] == 'Rapid Pink']['quantidade2']
e2 = dados[dados['centro_distribuicao'] == 'Grãos Blue']['quantidade2']
e3 = dados[dados['centro_distribuicao'] == 'Gold Beach']['quantidade2']
e4 = dados[dados['centro_distribuicao'] == 'Papa Léguas']['quantidade2']
e5 = dados[dados['centro_distribuicao'] == 'Tree True']['quantidade2']

print(st.f_oneway(e1,
            e2,
            e3,
            e4,
            e5))

grupo2 = dados[dados['quantidade2'] <= 4]['centro_distribuicao']
print(grupo2)

quantidade = dados['quantidade2']

lista = []

for i in quantidade:
    if i > 4:
        lista.append(1)
    else:
        lista.append(2)

#print(lista)

dados['nova_coluna'] = lista

e1 = dados[dados['centro_distribuicao'] == 'Rapid Pink']['nova_coluna']
e2 = dados[dados['centro_distribuicao'] == 'Grãos Blue']['nova_coluna']
e3 = dados[dados['centro_distribuicao'] == 'Gold Beach']['nova_coluna']
e4 = dados[dados['centro_distribuicao'] == 'Papa Léguas']['nova_coluna']
e5 = dados[dados['centro_distribuicao'] == 'Tree True']['nova_coluna']

print(st.f_oneway(e1,
            e2,
            e3,
            e4,
            e5))

print(pg.anova(dv= 'quantidade2',
         between= 'produto',
         data = dados,
         detailed= True))

print(novo)

upper_limit = dados['quantidade2'].mean() + 3 * dados['quantidade2'].std()
print(upper_limit)

lower_limit = dados['quantidade2'].mean() - 3 * dados['quantidade2'].std()
print(lower_limit)

print(dados[(dados['quantidade2'] > upper_limit) | (dados['quantidade2'] < lower_limit)])
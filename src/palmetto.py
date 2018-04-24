from palmettopy.palmetto import Palmetto
palmetto = Palmetto()
words = []
score = palmetto.get_coherence(words)
score = round(score,3)
print (score)
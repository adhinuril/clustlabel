from palmettopy.palmetto import Palmetto
palmetto = Palmetto()
words = ["connection","network"]
score = palmetto.get_coherence(words)
score = round(score,3)
print (score)
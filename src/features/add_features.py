"""Add  new features."""

class features():
    def add_connection(data):
        def update_fit_score(score,connection):
            fit_score = score - (1/(connection+10))
            return fit_score

        def add_connection_to_score(data):
            data['Jaccard Score'] = data.apply(lambda x: update_fit_score(x['Jaccard Score'],x['connection']),axis=1)
            data['GloVe Score'] = data.apply(lambda x: update_fit_score(x['GloVe Score'],x['connection']),axis=1)
            data['Doc2Vec Score'] = data.apply(lambda x: update_fit_score(x['Doc2Vec Score'],x['connection']),axis=1)
            data['BERT Score'] = data.apply(lambda x: update_fit_score(x['BERT Score'],x['connection']),axis=1)
            return data
        return data
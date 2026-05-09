"""Model B: Distractor and Hint Generation"""

import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker')
    try:
        nltk.data.find('chunkers/maxent_ne_chunker_tab')
    except LookupError:
        nltk.download('maxent_ne_chunker_tab')
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

download_nltk_data()

class DistractorGenerator:
    def __init__(self, ranker_model=None):
        self.ranker_model = ranker_model

    def generate_distractors_race(self, original_options, correct_answer):
        """Mode 1: RACE sample mode (returns original wrong options)."""
        distractors = [opt for opt in original_options if opt != correct_answer]
        return distractors[:3]

    def _extract_candidates(self, passage):
        """Extract distractor candidates from a custom passage."""
        words = nltk.word_tokenize(passage)
        pos_tags = nltk.pos_tag(words)
        
        candidates = set()
        
        # 1. Important/Frequent content words (nouns, verbs, adjectives)
        content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}
        content_words = [w for w, tag in pos_tags if tag in content_tags and len(w) > 3]
        candidates.update(content_words)
        
        # 2. Capitalized words, Dates, Numbers (NER)
        chunks = nltk.ne_chunk(pos_tags)
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity = ' '.join(c[0] for c in chunk)
                candidates.add(entity)
        
        # 3. Numbers
        numbers = [w for w, tag in pos_tags if tag == 'CD']
        candidates.update(numbers)
        
        return list(candidates)

    def _extract_features(self, candidate, correct_answer, question, passage):
        """Calculate feature engineering values for distractor ranking."""
        features = {}
        
        candidate_lower = candidate.lower()
        correct_lower = correct_answer.lower()
        passage_lower = passage.lower()
        question_lower = question.lower()
        
        # Frequency in passage
        features['candidate_frequency'] = passage_lower.count(candidate_lower)
        
        # Length
        features['candidate_length'] = len(candidate)
        
        # Similarity to correct answer (character level overlap)
        cand_chars = set(candidate_lower)
        corr_chars = set(correct_lower)
        overlap = len(cand_chars.intersection(corr_chars))
        features['candidate_answer_similarity'] = overlap / max(1, len(corr_chars))
        
        # Similarity to question (word level overlap)
        cand_words = set(nltk.word_tokenize(candidate_lower))
        ques_words = set(nltk.word_tokenize(question_lower))
        q_overlap = len(cand_words.intersection(ques_words))
        features['candidate_question_similarity'] = q_overlap / max(1, len(ques_words))
        
        # Is correct answer exact match?
        features['is_correct'] = 1 if candidate_lower == correct_lower else 0
        
        # Create an array of features (order must match training)
        feature_array = [
            features['candidate_frequency'],
            features['candidate_length'],
            features['candidate_answer_similarity'],
            features['candidate_question_similarity'],
            features['is_correct']
        ]
        return feature_array

    def generate_distractors_custom(self, passage, question, correct_answer):
        """Mode 2: Custom passage mode."""
        candidates = self._extract_candidates(passage)
        
        if not candidates:
            return ["None", "None", "None"]
            
        if self.ranker_model is None:
            # Fallback if no ranker is loaded: random sample or first few
            distractors = [c for c in candidates if c.lower() != correct_answer.lower()]
            return distractors[:3]
            
        scored_candidates = []
        for cand in candidates:
            if cand.lower() == correct_answer.lower():
                continue
            feats = self._extract_features(cand, correct_answer, question, passage)
            # Assuming ranker returns probability of being a good distractor (class 1)
            score = self.ranker_model.predict_proba([feats])[0][1]
            scored_candidates.append((score, cand))
            
        # Sort by score descending
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Select top 3 diverse candidates
        selected = []
        for score, cand in scored_candidates:
            if cand not in selected:
                selected.append(cand)
            if len(selected) >= 3:
                break
                
        # Fill if not enough
        while len(selected) < 3:
            selected.append(f"Option {len(selected)+1}")
            
        return selected

class HintGenerator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def generate_hints(self, article, question, correct_answer):
        """Generates 3 graduated hints using sentence ranking."""
        sentences = nltk.sent_tokenize(article)
        if not sentences:
            return ["No article provided.", "No article provided.", "No article provided."]
            
        # TF-IDF vectors
        corpus = sentences + [question]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            # Vocabulary empty
            return ["Look at the text.", "Think about the topic.", f"The answer is related to {correct_answer}"]
            
        # Compute cosine similarity between question (last row) and sentences (all other rows)
        question_vector = tfidf_matrix[-1]
        sentence_vectors = tfidf_matrix[:-1]
        
        similarities = cosine_similarity(question_vector, sentence_vectors).flatten()
        
        # Rank sentences by similarity
        ranked_indices = similarities.argsort()[::-1]
        top_sentence = sentences[ranked_indices[0]] if len(ranked_indices) > 0 else "Look closely at the passage."
        
        # Hint 1: general clue (topic of the most relevant sentence)
        hint1 = f"Look for the sentence that talks about: {' '.join(nltk.word_tokenize(top_sentence)[:5])}..."
        
        # Hint 2: more specific clue (relevant sentence, but hide answer)
        # Try to replace exact match or word overlap of answer
        if correct_answer.lower() in top_sentence.lower():
            pattern = re.compile(re.escape(correct_answer), re.IGNORECASE)
            hint2 = pattern.sub("_____", top_sentence)
        else:
            hint2 = top_sentence + " (The answer is related to this)."
            
        # Hint 3: near-explicit clue
        hint3 = f"The answer is '{correct_answer[:len(correct_answer)//2 + 1]}...' and it's found in the sentence: {hint2}"
        
        return [hint1, hint2, hint3]

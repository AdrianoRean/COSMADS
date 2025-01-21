import re
from copy import deepcopy
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

model = None

def bert_cosine_similarity(sentence1, sentence2):
    global model
    global tokenizer
    
    # Tokenize the sentences
    tokens1 = tokenizer.tokenize(sentence1)
    tokens2 = tokenizer.tokenize(sentence2)
    
    # Convert tokens to input IDs
    input_ids1 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)  # Batch size 1
    input_ids2 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)  # Batch size 1

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs1 = model(input_ids1)
        outputs2 = model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token

    # Calculate similarity
    similarity_score = cosine_similarity(embeddings1, embeddings2)
    return similarity_score[0][0]

def correct_obvious_word_mistake(function_to_change, call_keys = [], hard_coded_words = [], similarity_treshold = 0.85):
    global model
    global tokenizer
    function_changed = deepcopy(function_to_change)
    
    #find all call keys in the function
    call_regex_string = re.compile(r"""
                                        call\( #anything inside a call()
                                        (.*)
                                       
                                       \)
                                   """, re.VERBOSE)
    
    call_matches = list(set(call_regex_string.findall(function_changed)))  
    
    call_match_dict = {} 
    inside_call_matches = []
    
    inside_call_regex_string = re.compile(r"""
                                          \b
                                          (\w+)
                                            \s*
                                            =
                                          """, re.VERBOSE)
    for index, match in enumerate(call_matches):
        temp_inside_matches = inside_call_regex_string.findall(match)
        inside_call_matches = inside_call_matches + temp_inside_matches
        for inside_match in temp_inside_matches:
            if inside_match not in call_match_dict:
                call_match_dict[inside_match] = []
            call_match_dict[inside_match].append(index)
    
    #check for each possible value if there are matches
    for word in call_keys:
        best_matches = []
        for call_match in inside_call_matches:
            similarity = bert_cosine_similarity(word, call_match)
            if similarity >= similarity_treshold:
                best_matches.append((call_match, similarity))
        print(f"Key: {word} - Matches: {best_matches}")
        for match in best_matches:
            if match[1] == 1: #perfect match
                if word == match[0]: #also cases are equal, no need to change
                    continue
            #imperfect match or case difference, probable typo mistake
            calls_indexes = call_match_dict[match[0]] #find calls to be replaced
            for call_index in calls_indexes:
                call = call_matches[call_index] #retrieve single call
                print(f"Key: {word} - calls : {call}")
                new_call = call.replace(match[0], word) #fix it
                call_matches[call_index] = new_call #save bettered version for future use
                function_changed = function_changed.replace(call, new_call) #replace in actual function
                    
    #find all hard coded words in function
    hard_coded_regex_string = re.compile(r"\"([\w+\s]+)\"")
    hard_coded_matches = list(set(hard_coded_regex_string.findall(function_changed)))  
    print(hard_coded_matches)  
    #check for each possible value if there are matches
    for word in hard_coded_words:
        best_matches = []
        for hard_match in hard_coded_words:
            similarity = bert_cosine_similarity(word, hard_match)
            if similarity >= similarity_treshold:
                best_matches.append((hard_match, similarity))
        print(f"Word: {word} - Matches: {best_matches}")
        for match in best_matches:
            if match[1] == 1: #perfect match
                if word == match[0]: #also cases are equal, no need to change
                    continue
            #imperfect match or case difference, probable typo mistake
            specific_hard_coded_regex = re.compile(f"\"{match[0]}\"")
            function_changed = specific_hard_coded_regex.sub(f'"{word}"', function_changed)
            
    return function_changed

# Example usage
if __name__ == "__main__":
    FUNCTION_TO_CHECK = """def example_function():
    greeting = "Hello, World!"
    name = "Alice"
    positionIds = blablabla.call(Position = ("I'm your friend", "EQUAL"), puppafava=(pippobaudo))
    locationIds = blablabla.call(location = ("Activist", "Minor"), puppafava=(pippobaudo))
    pippeIds = blablabla.call(prugna = ("greade", "Minor"), puppafava=(mulino))
    mecojons = brururur.call(gregre = ("gragra", "EQUAL"), Cognomi=(pippobaudo))
    print(f"{greeting}, {name}!")
    return "Done" """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print(correct_obvious_word_mistake(FUNCTION_TO_CHECK, ["position", "puppafave", "cognome"], ["Gnappetta", "Gral"], similarity_treshold=0.5))

    '''strings, matches = extract_hardcoded_strings(FUNCTION_TO_CHECK)
    
    to_match = "positionID"
    
    match = process.extractBests(to_match, choices=strings+matches, score_cutoff=80)
    print("Hard-coded strings:", strings)
    print("Call strings:", matches)
    print("Matches:", set(match))
    
    #key_regex = re.compile(r"call\(.*\b(.*)\b\s*=.*\)")
    #key_matches = list(set(key_regex.findall(FUNCTION_TO_CHECK)))
    
    #find key in call to change it
    p = FUNCTION_TO_CHECK
    
            
    print(p)'''
    
    '''
    matches = process.extractBests("amo", choices=["amo", "Amo", "amO", "AMO", "amor"], processor=custom_processor)
    for match in matches:
        if match[1] == 100:
            if match[0] == "amo":
                print(f"perfect match: {match[0]}")
            else:
                print(f"case difference: {match[0]}")
        else:
            print(f"typo: {match[0]}")
    '''
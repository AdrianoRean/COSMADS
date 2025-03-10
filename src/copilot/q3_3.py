import json
import numpy as np

# Assuming the presence of a function or method to determine if a cardboard is defective or not
# This is a placeholder function. You should replace it with the actual implementation.
def is_defective(cardboard_id: int) -> bool:
    # Placeholder logic for determining if a cardboard is defective
    # Replace this with the actual logic
    return np.random.choice([True, False])

def generate_cardboard_data(diecutter_id: int, num_cardboards: int = 30):
    defect_free_count = 0
    defective_count = 0
    
    for _ in range(num_cardboards):
        # Generate a unique cardboard_id for simplicity, replace with actual id generation logic
        cardboard_id = np.random.randint(1000, 9999)
        
        if is_defective(cardboard_id):
            defective_count += 1
        else:
            defect_free_count += 1
    
    return {
        "defect_free_count": defect_free_count,
        "defective_count": defective_count
    }

# Generate data for diecutter with id 7
data = generate_cardboard_data(diecutter_id=7)

# Assuming we want to generate a list of dictionaries for each diecutter, but since the query specifies a single diecutter,
# we wrap the data in a list to match the requested format.
data_list = [data]

# Write the data to a JSON file
with open('cardboard_data.json', 'w') as f:
    json.dump(data_list, f)

print("Data written to cardboard_data.json")
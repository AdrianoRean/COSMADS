import os

if __name__ == '__main__':

    databases = [
        "craftbeer",
        "trains",
        "music_tracker",
        "citeseer",
        "genes",
        "human_resources",
        "cookbook",
        "computer_student",
        "college_completion",
        "software_company",
        "shipping",
        "cs_semester",
        "law_episode",
        "language_corpus",
        "image_and_language",
        "movielens",
        "chicago_crime",
        "simpson_episodes",
        "car_retails",
        "video_games",
        "retails",
        #"address",
        "professional_basketball",
        "student_loan",
        "book_publishing_company",
        "olympics",
        #"synthea",
        #"talkingdata",
        "books",
        "public_review_platform",
        "movie_3",
        "movies_4",
        "soccer_2016",
        #"hockey",
        "mondial_geo",
        #"works_cycles"
    ]

    # models = [
    #     "openai",
    #     # "mistral",
    #     # "anthropic"
    #     #"deepseek",
    # ]
    model = "anthropic"

    for database in databases:
        print(f"Running experiments for database {database} and model {model}")
        result = os.system(f"python -u run_evaluation.py --database {database} --model {model}")
        if result != 0:
            print(f"Error running experiments for database {database} and model {model}")
        else:
            print(f"Experiments for database {database} and model {model} completed successfully")
            


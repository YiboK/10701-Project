## Append to original Yelp data with text from yelp_sarcasm_review_500.txt
## Data entry: text: line from file, starts: 1, other columns: blank
import pandas as pd

def main():
    yelp_file = "dataset/yelp.csv"
    yelp_data = pd.read_csv(yelp_file)

    append_file = "dataset/yelp_sarcasm_review_500.txt"

    # transform appended text to Yelp review data entries
    append_data = []
    with open(append_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line:  # Skip empty lines
                append_data.append({"text": line, "stars": 1})
    append_df = pd.DataFrame(append_data)

    # generated appended Yelp with orginal Yelp data and appended texts
    combined_df = pd.concat([yelp_data, append_df], ignore_index=True)

    # save the combined data into a new CSV file
    output_file = "dataset/yelp_appended.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"Combined data saved to {output_file}")

if __name__ == "__main__":
    main()

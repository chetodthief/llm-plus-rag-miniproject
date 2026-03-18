import pandas as pd

try:
    df = pd.read_csv(r"D:\llm\llm-project\wiki_movie_plots_deduped\wiki_movie_plots_deduped.csv")
    wick = df[df['Title'].str.contains('John Wick', case=False, na=False)]
    print(f"John Wick entries: {len(wick)}")
    if len(wick) > 0:
        print(wick[['Title', 'Release Year']])
except Exception as e:
    print(e)

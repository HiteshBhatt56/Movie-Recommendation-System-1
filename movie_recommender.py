"""
Movie Recommendation System
---------------------------
Content-based recommendation using TF-IDF and cosine similarity.

Run:
    pip install -r requirements.txt
    python movie_recommender.py --movie "Inception"

List movies:
    python movie_recommender.py --list-movies
"""

import argparse
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIES = [
    {"title":"The Dark Knight","genres":"Action Crime Drama Superhero Batman","overview":"A masked hero protects a city from a criminal mastermind."},
    {"title":"Inception","genres":"Action Sci-Fi Thriller Mind-Bending","overview":"A skilled thief enters dreams to perform a dangerous mission."},
    {"title":"Interstellar","genres":"Adventure Drama Sci-Fi Space Exploration","overview":"Explorers travel through space to find a new home for humanity."},
    {"title":"Pulp Fiction","genres":"Crime Drama Thriller Cult Classic","overview":"Several interconnected stories unfold in the Los Angeles criminal underworld."},
    {"title":"The Matrix","genres":"Action Sci-Fi Cyberpunk Artificial Intelligence","overview":"A hacker discovers that reality is a simulated world controlled by machines."},
    {"title":"Superbad","genres":"Comedy Teen High-School Friendship","overview":"Two friends try to make the most of their final days of high school."},
    {"title":"The Hangover","genres":"Comedy Las Vegas Friendship Adventure","overview":"Friends search for clues after a wild night in Las Vegas."},
    {"title":"Se7en","genres":"Crime Drama Mystery Thriller Serial Killer","overview":"Two detectives investigate a series of disturbing crimes."},
    {"title":"Avatar","genres":"Action Adventure Fantasy Sci-Fi Alien","overview":"A marine becomes involved in a conflict on an alien world."},
    {"title":"Jurassic Park","genres":"Adventure Sci-Fi Thriller Dinosaurs","overview":"Visitors struggle to survive after cloned dinosaurs escape."},
    {"title":"Star Wars","genres":"Action Adventure Fantasy Sci-Fi Space","overview":"A young hero joins a rebellion against a powerful empire."},
    {"title":"The Lord of the Rings","genres":"Adventure Drama Fantasy Epic","overview":"A group of companions journeys to destroy a powerful ring."},
    {"title":"Iron Man","genres":"Action Adventure Sci-Fi Superhero Technology","overview":"An inventor develops advanced technology and becomes a superhero."},
    {"title":"Avengers: Endgame","genres":"Action Adventure Drama Sci-Fi Superhero","overview":"A team of heroes attempts to undo a devastating event."},
    {"title":"Toy Story","genres":"Animation Adventure Comedy Family Friendship","overview":"Living toys discover friendship and learn to work together."},
    {"title":"Finding Nemo","genres":"Animation Adventure Comedy Family Ocean","overview":"A father searches the ocean for his missing son."},
    {"title":"Titanic","genres":"Drama Romance Historical","overview":"Two people from different backgrounds fall in love during a historic voyage."},
    {"title":"The Notebook","genres":"Drama Romance Love","overview":"A couple's enduring relationship is tested by time and circumstances."},
    {"title":"La La Land","genres":"Comedy Drama Music Romance","overview":"Two aspiring artists pursue their dreams while building a relationship."},
    {"title":"The Shawshank Redemption","genres":"Drama Prison Friendship Hope","overview":"A prisoner maintains friendship and hope through difficult years."},
    {"title":"Forrest Gump","genres":"Drama Romance History","overview":"A kind-hearted man experiences extraordinary moments throughout his life."},
    {"title":"The Pursuit of Happyness","genres":"Biography Drama Family","overview":"A father works toward a better life for himself and his son."},
    {"title":"Moneyball","genres":"Biography Drama Sport Baseball","overview":"A baseball manager uses data-driven methods to build a competitive team."},
    {"title":"Rocky","genres":"Drama Sport Boxing","overview":"An underdog boxer gets an unexpected opportunity to compete."},
    {"title":"Mad Max: Fury Road","genres":"Action Adventure Sci-Fi Thriller","overview":"Survivors cross a dangerous wasteland while escaping a tyrant."},
    {"title":"John Wick","genres":"Action Crime Thriller Assassin","overview":"A retired assassin is drawn back into a dangerous criminal world."},
    {"title":"Get Out","genres":"Horror Mystery Thriller","overview":"A visitor uncovers disturbing secrets while visiting his partner's family."},
    {"title":"A Quiet Place","genres":"Drama Horror Sci-Fi Thriller Family","overview":"A family survives while avoiding creatures that hunt by sound."},
]

def load_movies(csv_path=None):
    if not csv_path:
        return pd.DataFrame(MOVIES)
    df = pd.read_csv(csv_path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "title" not in cols or "genres" not in cols:
        raise ValueError("CSV must contain 'title' and 'genres' columns.")
    rename = {cols["title"]: "title", cols["genres"]: "genres"}
    for col in ("overview", "cast", "director"):
        if col in cols:
            rename[cols[col]] = col
    df = df.rename(columns=rename)
    for col in ("overview", "cast", "director"):
        if col not in df:
            df[col] = ""
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["genres"] = df["genres"].fillna("").astype(str)
    df = df[df["title"] != ""].drop_duplicates("title").reset_index(drop=True)
    return df

class MovieRecommender:
    def __init__(self, movies):
        self.movies = movies.copy()
        text_cols = [c for c in ("genres","overview","cast","director") if c in self.movies]
        self.movies["features"] = self.movies[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.matrix = self.vectorizer.fit_transform(self.movies["features"])
        self.similarity = cosine_similarity(self.matrix)

    def find_index(self, title):
        query = title.strip().lower()
        exact = self.movies[self.movies["title"].str.lower() == query]
        if not exact.empty:
            return exact.index[0]
        partial = self.movies[self.movies["title"].str.lower().str.contains(query, regex=False)]
        if not partial.empty:
            return partial.index[0]
        return None

    def recommend(self, title, n=5):
        idx = self.find_index(title)
        if idx is None:
            return pd.DataFrame(columns=["title","genres","similarity"])
        scores = sorted(enumerate(self.similarity[idx]), key=lambda x:x[1], reverse=True)
        rows = []
        for i, score in scores:
            if i == idx:
                continue
            rows.append({"title":self.movies.iloc[i]["title"],
                         "genres":self.movies.iloc[i]["genres"],
                         "similarity":float(score)})
            if len(rows) >= n:
                break
        return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser(description="Content-based movie recommendation system.")
    parser.add_argument("--movie", help="Movie title to use for recommendations.")
    parser.add_argument("--n", type=int, default=5, help="Number of recommendations.")
    parser.add_argument("--csv", help="Optional CSV containing title and genres columns.")
    parser.add_argument("--list-movies", action="store_true", help="List available movies.")
    args=parser.parse_args()
    if args.n < 1 or args.n > 20:
        print("ERROR: --n must be between 1 and 20.", file=sys.stderr); sys.exit(1)
    try:
        movies=load_movies(args.csv)
        engine=MovieRecommender(movies)
        if args.list_movies:
            print("\n".join(movies["title"])); return
        if not args.movie:
            print('Example: python movie_recommender.py --movie "Inception"'); return
        idx=engine.find_index(args.movie)
        if idx is None:
            print(f"Movie not found: {args.movie}"); return
        result=engine.recommend(args.movie,args.n)
        print(f"\nRecommendations for: {movies.iloc[idx]['title']}\n")
        for i,row in result.iterrows():
            print(f"{i+1}. {row['title']} | {row['genres']} | similarity={row['similarity']:.3f}")
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()

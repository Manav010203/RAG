#!/usr/bin/env python3

import argparse
import math

from lib.keyword_search import build_command, search_command,InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    

    tf_parser = subparsers.add_parser("tf", help="Get term frequency in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")\
    
    idf_parser = subparsers.add_parser("idf",help="finding the idf score of a term")
    idf_parser.add_argument("term",type=str,help="term for which to calculate the idf")

    tfidf_parser = subparsers.add_parser("tfidf",help="finding tf-idf")
    tfidf_parser.add_argument("doc_id",type=str,help="document id")
    tfidf_parser.add_argument("term",type=str,help="term to find tf idf for")

    args = parser.parse_args()
    
    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(tf)
        case "idf":
            term = args.term
            index = InvertedIndex()
            index.load()
            occ = index.get_idf(term)
            idf = math.log((len(index.docmap)+1) / (occ+1))
            # print(idf)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        # case "tfidf":
        #     term = args.term
        #     doc_id = args.doc_id
        #     index = InvertedIndex()
        #     index.load()
        #     tf = index.get_tf(args.doc_id, args.term)
        #     occ = index.get_idf(term)
        #     idf = math.log((len(index.docmap)+1) / (occ+1))
        #     tf_idf = tf * idf
        #     print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

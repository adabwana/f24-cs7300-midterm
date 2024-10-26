(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []
  (clay/make!
   {:format              [:quarto :html]
    :book                {:title "CS7300: Practical Midterm"}
    :subdirs-to-sync     ["notebooks" "data" "images"]
    :source-path         ["src/index.clj"
                          "notebooks/question_1.ipynb"
                          "notebooks/question_2.ipynb"
                          "notebooks/technical_report.md"]
    :base-target-path    "docs"
    :clean-up-target-dir true}))

(comment
  (build))

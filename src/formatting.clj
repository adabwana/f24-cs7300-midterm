(ns formatting
  (:require
   [scicloj.kindly.v4.api :as kindly]
   [scicloj.kindly.v4.kind :as kind]))

; ## Utils

(def md (comp kindly/hide-code kind/md))
(def question (fn [content] ((comp kindly/hide-code kind/md) (str "## " content "\n---"))))
(def sub-question (fn [content] ((comp kindly/hide-code kind/md) (str "#### *" content "*"))))
(def sub-sub (fn [content] ((comp kindly/hide-code kind/md) (str "***" content "***"))))
(def answer (fn [content] (kind/md (str "> <span style=\"color: black; font-size: 1.5em;\">**" content "**</span>"))))
(def formula (comp kindly/hide-code kind/tex))
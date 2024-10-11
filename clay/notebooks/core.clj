(ns notebooks.core
  (:require [scicloj.clay.v2.api :as clay]))

(comment

  ;;
  (clay/make! {:source-path "clay/notebooks/core.clj"})

  (+ 1 1)
  (clay/make! {:single-value
               (+ 1 1)})
  ;
  )

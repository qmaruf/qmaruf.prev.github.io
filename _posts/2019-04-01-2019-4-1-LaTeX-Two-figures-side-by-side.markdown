---
title: Latex Two Figures Side By Side
date: 2019-04-01 00:00:00 Z
---

```
\begin{figure}
    \centering
    \begin{subfigure}{0.4\textwidth}
    \includegraphics[width=\textwidth, height=3cm]{figures/1.png}
    \caption{subfigure 1 caption}
    \end{subfigure}
    ~
    \begin{subfigure}{0.4\textwidth}
    \includegraphics[width=\textwidth, height=3cm]{figures/2.png}
    \caption{subfigure 2 caption}
    \end{subfigure}
    \caption{figure caption}
\end{figure}
```

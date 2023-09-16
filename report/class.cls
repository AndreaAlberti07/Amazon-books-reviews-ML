\newif\if@suppmat\@suppmattrue
\DeclareOption{nosupp}{\@suppmatfalse}
\ExecuteOptions{nosupp}
\ProcessOptions\relax
\LoadClass{article}

\RequirePackage{etoolbox,xpatch}
\RequirePackage{ifxetex,ifluatex}
\RequirePackage{setspace}
\RequirePackage[table]{xcolor}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{rotating}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage[style]{abstract}
\RequirePackage[marginal]{footmisc}
\RequirePackage{caption,subcaption}
\RequirePackage{enumitem}
\RequirePackage{authblk}
\RequirePackage{textcomp}
\RequirePackage[absolute]{textpos}
\RequirePackage[right]{lineno}
\RequirePackage{amsmath}

\def\@keywordsname{Keywords}
\def\@authornotename{Author's notes}
\def\@corrauthorname{Corresponding author}
\def\@editorname{Edited by}
\def\@runningauthor{}
\def\@corrauthor{}
\def\@authornote{}
\definecolor{PANLightBlue}{RGB}{105,178,218}
\definecolor{PANDarkBlue}{RGB}{19,126,194}
\definecolor{PANMaroon}{RGB}{105,27,46}
\definecolor{PANDarkGray}{RGB}{153,153,153}

\RequirePackage{microtype}
\newif\ifxetexorluatex
\ifxetex
  \xetexorluatextrue
\else
  \ifluatex
    \xetexorluatextrue
  \else
    \xetexorluatexfalse
  \fi
\fi

\ifxetexorluatex
  \RequirePackage[no-math]{fontspec}
  \let\boldsymbol\symbf
  % Patch microtype for older versions; from https://tex.stackexchange.com/a/373607/226
  \@ifpackagelater{microtype}{2017/07/05}{}{%
    \patchcmd{\MT@is@composite}
    {\expandafter\expandafter\expandafter}
    {\ifx\UnicodeEncodingName\@undefined\else
        \expandafter\expandafter\expandafter\MT@is@uni@comp\MT@char\iffontchar\else\fi\relax
      \fi\expandafter\expandafter\expandafter}
    {}{}
    \def\MT@is@uni@comp#1\iffontchar#2\else#3\fi\relax{%
      \ifx\\#2\\\else\edef\MT@char{\iffontchar#2\fi}\fi
    }
  }
\else
  \RequirePackage[utf8]{inputenc}
  \RequirePackage[T1]{fontenc}
\fi
\RequirePackage[default]{sourcesanspro}
\RequirePackage[nosymbolsc,slantedGreek]{newtxsf}

\setstretch{1.15}

\RequirePackage{geometry}
\geometry{paperwidth=8.25in,
  paperheight=11in,
  left=2.1cm,
  top=3cm,
  bottom=2.4cm,
  right=2.1cm,
  footskip=2\baselineskip,
  footnotesep=\baselineskip,
  headheight=21\p@,
  headsep=\baselineskip
}

\newcommand{\runningauthor}[1]{\def\@runningauthor{#1}}
\newcommand{\shortname}[1]{\def\@shortname{#1}}
\newcommand{\authornote}[1]{\def\@authornote{#1}}
\newcommand{\corrauthor}[1]{\def\@corrauthor{#1}}
\newcommand{\course}[1]{\def\@course{#1}}
\newcommand{\githubrepo}[1]{\def\@githubrepo{#1}}




\renewcommand{\Authfont}{\large}
\renewcommand{\Affilfont}{\small\itshape\mdseries}
\renewcommand\AB@authnote[1]{\textsuperscript{#1}}
\renewcommand\AB@affilnote[1]{\textsuperscript{#1}}
\setlength{\affilsep}{\baselineskip}

\newcommand{\@cover@logo}{%
  \begin{textblock*}{4cm}[1,0](\dimexpr\Gm@lmargin-1cm,\Gm@tmargin)
    \fontsize{60\p@}{60\p@}\selectfont\raggedleft%
    \textcolor{PANMaroon}{PA}
  \end{textblock*}
}
\renewcommand{\@maketitle}{%
  %\@cover@logo%
  \noindent\begin{minipage}{\textwidth}
    \raggedright\setstretch{1.0}
    {\centering\LARGE\bfseries\@title\par}
    \centering\vskip\baselineskip
    \@author\par
    \vskip\baselineskip
    {\small{Github repository: \url{\@githubrepo}}\par}
  \end{minipage}\bigskip\normalfont\normalsize%
}
\appto{\maketitle}{%
  \ifdefvoid{\@authornote}{}{\begingroup
    \renewcommand{\thefootnote}{}
    \footnotetext{\emph{\@authornotename:} \@authornote}%
    \endgroup}%
}

\renewcommand{\abstitlestyle}[1]{{\large\bfseries #1\par}}
\setlength{\absleftindent}{\z@}
\setlength{\absrightindent}{\absleftindent}
\preto{\abstract}{%
  \setlength{\parindent}{\z@}%
  \newcommand{\keywords}[1]{%
    \emph{\textbf{\@keywordsname ---} }\emph{#1}\par
  }
  {\textcolor{PANDarkBlue}{\rule{\textwidth}{1.5\p@}}}\\%
}
\appto{\abstract}{%
  \setlength{\parindent}{\z@}%
  \setlength{\parskip}{1em}%
  \vskip-\parskip%
  \noindent\ignorespaces%
}
\appto{\endabstract}{%
  {\raisebox{\baselineskip}{\textcolor{PANDarkBlue}{\rule{\textwidth}{1\p@}}}}%
}

\titleformat{\section}{\Large\bfseries}{\hspace*{1em}\makebox[\z@][r]{\thesection}}{1em}{}
\titleformat{\subsection}{\large\bfseries}{\hspace*{1em}\makebox[\z@][r]{\thesubsection}}{1em}{}
\titlespacing*{\section}{\z@}{\baselineskip}{\z@}
\titlespacing*{\subsection}{\z@}{\baselineskip}{\z@}
\titlespacing*{\subsubsection}{\z@}{\baselineskip}{\z@}
\setcounter{secnumdepth}{2}

\setlist{leftmargin=*,nosep}

\setlength{\footnotemargin}{-1em}
\pretocmd{\@makefntext}{%
  \renewcommand{\@makefnmark}{\thefootnote}
}{}{}

\fancyhf{}
\renewcommand{\footrulewidth}{1.5\p@}
\renewcommand{\headrule}{}
\preto{\footrule}{\begingroup\color{PANLightBlue}}
\appto{\footrule}{\endgroup}
%\fancyhead[L]{\hspace*{-4cm}\Huge\color{PANMaroon}PA}
\fancyfoot[R]{\small\thepage}
\fancyhead[L]{}
\fancyhead[C]{\small\emph{\@shortname}}

\pagestyle{fancy}
\fancypagestyle{plain}{%
  \fancyhf{}
  \renewcommand{\footrule}{}
  \fancyfoot[R]{\small\thepage}
}

\captionsetup*{singlelinecheck=false,font={small},labelfont={bf,color=PANDarkBlue},labelsep=period}
\captionsetup*[table]{position=top}
\captionsetup*[sub]{labelfont=small,justification=centering}
\newcommand{\headrow}{\rowcolor{PANDarkGray}}
\newcommand{\thead}[1]{\bfseries #1\rule[-1.2ex]{0pt}{2em}}
\renewcommand{\arraystretch}{1.2}

\RequirePackage[backend=biber,authordate,giveninits=true,uniquename=mininit,natbib]{biblatex-chicago}
\xpatchbibmacro{name:andothers}{%
  \bibstring{andothers}%
}{%
  \bibstring[\emph]{andothers}%
}{}{}

\newcommand{\insert@supplmat}{\section*{Supplementary Material}(This is dummy text) For supplementary material accompanying this paper, please visit \url{https://doi.org/10.1017/pan.xxxx.xx.}}
\if@suppmat
  \xpretocmd{\printbibliography}{\insert@supplmat}{}{}
  \AtBeginEnvironment{thebibliography}{\insert@supplmat}
\fi
\renewcommand{\bibfont}{\small}

\PassOptionsToPackage{hyphens}{url}
\RequirePackage{hyperref}
\hypersetup{colorlinks,allcolors=PANDarkBlue}
\urlstyle{same}

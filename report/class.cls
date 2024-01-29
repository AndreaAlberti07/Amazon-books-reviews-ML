\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{class}[2019/10/29 AJCEAM paper class]

% Initial Statements
\DeclareOption{11pt}{\OptionNotUsed}
\DeclareOption{12pt}{\OptionNotUsed}
\DeclareOption{onecolumn}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\newif\if@isblind\@isblindfalse
\DeclareOption{blind}{\@isblindtrue}

\ProcessOptions\relax
\LoadClass[twoside,a4paper,10pt,twocolumn,english]{article}



% Required Packages
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{multirow}
\usepackage[T1]{fontenc}
\usepackage{ae,aecompl}
\usepackage{pslatex}
\usepackage[samesize]{cancel}
\usepackage{graphicx}
\usepackage{setspace}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{csquotes}
\usepackage{amssymb}
\usepackage[table]{xcolor}
\usepackage{float}
\usepackage{mathptmx} % Use the Adobe Times Roman as the default text font together with math symbols from the Sym­bol, Chancery and Com­puter Modern fonts
\usepackage{url}
\Urlmuskip=0mu plus 1mu % break url lines
\usepackage{calc}
\usepackage{tabto}
\usepackage{lettrine}
\usepackage{listings}
\usepackage{subcaption}
\usepackage{biblatex}
%%% PDF file setup
\usepackage[bookmarks,
  bookmarksopen = true,
  bookmarksnumbered = true,
  breaklinks = true,
  colorlinks = true,
  linkcolor = black,
  urlcolor  = blue,
  citecolor = black,
  anchorcolor = green,
  hyperindex = true,
  hyperfigures
]{hyperref}

\lstset{
  basicstyle=\ttfamily\scriptsize,   % the size and font of the code
  numbers=none,                   % where to put the line numbers
  numberstyle=\scriptsize\color{gray},  % the style of the line numbers
  stepnumber=1,                   % the step between two line numbers
  numbersep=5pt,                  % how far the line numbers are from the code
  backgroundcolor=\color{white},  % choose the background color of the code
  showspaces=false,               % show spaces adding particular underscores
  showstringspaces=false,         % underline spaces within strings
  showtabs=false,                 % show tabs within strings adding particular underscores
  frame=single,                   % adds a frame around the code
  tabsize=1,                      % sets default tabsize to 2 spaces
  captionpos=b,                   % sets the caption-position to bottom
  breaklines=true,                % sets automatic line breaking
  breakatwhitespace=false,        % sets if automatic breaks should only happen at whitespace
  escapeinside={\%*}{*)},
  frame=single,
  frameround=fttf,
  framexleftmargin=0mm,
  rulecolor=\color{white},
}
% Page setup
\setlength{\hoffset}{-1in}
\setlength{\voffset}{-1in}
\setlength{\oddsidemargin}{1.5cm}
\setlength{\evensidemargin}{1.5cm}
\setlength{\topmargin}{0.4cm}
\setlength{\headheight}{2.3cm}
\setlength{\headsep}{0.2cm}
\setlength{\textheight}{25.5 cm}
\setlength{\textwidth}{17.5cm}
\setlength{\footskip}{2,3 cm}
\setlength{\columnsep}{0.5cm}

% Author and Affiliate Setup
\usepackage[noblocks]{authblk}
\renewcommand\AB@affilsepx{\protect\par\protect\Affilfont}
\renewcommand\Authands{ and }
\renewcommand\Authand{ and }
\renewcommand{\Affilfont}{\fontsize{9}{12} \selectfont \itshape}
\renewcommand{\Authfont}{\fontsize{12}{14} \selectfont}

% Macro Declaration
\def\publicationdate#1{\gdef\@publicationdate{#1}}   % Publication Date
\def\abstract#1{\gdef\@englishabstract{#1}}        % Abstract in English
\def\github#1{\gdef\@github{#1}}
\def\keywords#1{\gdef\@englishkeywords{#1}}        % Keywords in English
\def\resumo#1{\gdef\@portugueseabstract{#1}}        % Resumen en inglés
\def\palavraschave#1{\gdef\@portuguesekeywords{#1}}        % Palabras clave en inglés

% Definition for contact data
\def\contactauthor#1{\gdef\@thecontactauthor{#1}}


% Header Definitions
\def\title#1{\gdef\@title{#1}\gdef\thetitle{#1}}
\def\titulo#1{\gdef\@titulo{#1}}
\def\shorttitle#1{\gdef\@theshorttitle{#1}}
\def\firstauthor#1{\gdef\@thefirstauthor{#1}}

\newcommand*{\authorheader}{%
  \begingroup
  \ifnum\value{authors}<4 % One author
    \if@isblind
      AUTHOR
    \else
      \MakeUppercase{\@thefirstauthor}
    \fi
  \else % 4 or more authors
    \if@isblind
      1st. AUTHOR~{\itshape et al.}
    \else
      {\MakeUppercase{\@thefirstauthor}~{\itshape et al.}}
    \fi
  \fi
  \endgroup
}
\newcommand{\vcenteredinclude}[1]{\begingroup
  \setbox0=\hbox{\includegraphics[height=\headheight]{#1}}%
  \parbox{\wd0}{\box0}\endgroup}


% Configuration of figure headings
\usepackage{caption}
\captionsetup[figure]{font=scriptsize,labelfont=bf,justification=centering}
\captionsetup[figure]{name=Fig.}
\DeclareCaptionFormat{myformat}{#1#2#3}%\rule{\linewidth}{.3pt}}
\captionsetup[figure]{format=myformat}

\captionsetup[table]{font=scriptsize,labelfont=bf,justification=centering}
\captionsetup[table]{name=Table}
\DeclareCaptionFormat{myformat}{#1#2#3}%\rule{\linewidth}{.3pt}}
\captionsetup[table]{format=myformat}

\usepackage{makecell} % break lines in cell tables
\renewcommand\theadalign{cc}
\renewcommand\theadgape{\Gape[4pt]}
\renewcommand\cellgape{\Gape[4pt]}
\addtolength{\belowcaptionskip}{-8pt}
\addtolength{\abovecaptionskip}{0pt}


% Work job cover
\newcommand*\varhrulefill[1][0.4pt]{\leavevmode\leaders\hrule height#1\hfill\kern0pt}
\renewcommand{\@maketitle}{%
  \null
  \onecolumn
  \begingroup
  \vspace{0cm}%
  {\begin{center}%
      \noindent\varhrulefill[0.25mm] \par
      {\fontsize{20}{24} \selectfont \@title \par}
      \noindent\varhrulefill[0.25mm] \par
      {\setlength{\tabcolsep}{0in}
        \if@isblind
        {\fontsize{12}{14} \selectfont Version for reviewers.} \\
        \else
        \begin{tabular}[c]{c}%
          {\@author}                                                 \\
          {\fontsize{8}{10} \selectfont Github page: \url{\@github}} \\
          {\fontsize{8}{10} \selectfont Date: \@publicationdate}     \\
        \end{tabular}
        \fi
      }
    \end{center}}% end of \parbox
  %\vspace{0.3cm}
  \noindent\varhrulefill[0.25mm] \par
  \vspace{0.4cm}
  \noindent {\fontsize{9}{12} \selectfont {\bfseries{Abstract }}---\@englishabstract} \par
  \vspace{0.3cm}
  \noindent {\fontsize{9}{12} \selectfont {\bfseries{Keywords }}---\@englishkeywords} \par
  \vspace{0.2cm}
  \noindent\varhrulefill[0.25mm]
  \twocolumn
  \endgroup
  %\vspace{0.5cm}
} % end of \@maketitle

%%% Header and footer settings
\usepackage{fancyhdr}
\usepackage{lastpage}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}
\fancypagestyle{FirstPage}{%
  \fancyhf{}
  \fancyhead[C]{\vcenteredinclude{figures/logo.png}}
  \fancyhead[L]{}
  \fancyhead[R]{}

}
\fancypagestyle{OtherPage}{%
  \fancyhf{}
  \fancyhead[R]{\fontsize{7}{10} \selectfont \thepage}
  \fancyhead[C]{{\fontsize{7}{10} \selectfont \MakeUppercase{\@theshorttitle}}}
  \fancyhead[L]{{\fontsize{7}{10} \selectfont \authorheader}}
}
% Section Format Settings
\renewcommand{\@seccntformat}[1]{\csname the#1\endcsname.~}

\renewcommand*{\thesection}{\arabic{section}}
\renewcommand{\section}{\@startsection{section}{1}{0mm}%
  {-12pt}%
  {6pt}%
  {\setlength\parindent{0em} \fontsize{12}{14} \selectfont \bfseries\textsc}}%
\renewcommand \thesubsection{\alph{subsection}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{0mm}%
  {-12pt}%
  {6pt}%
  {\setlength\parindent{0em} \fontsize{11}{13} \selectfont \bfseries}}%
\renewcommand \thesubsubsection{\arabic{subsubsection}}
\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{0mm}%
  {-12pt}%
  {6pt}%
  {\setlength\parindent{0em} \fontsize{11}{13} \selectfont \bfseries\textit}}



% Set bibliography style
\newcounter{iteminbib}
\addtocounter{iteminbib}{0}
\newlength{\bibmargin}
\setlength{\bibmargin}{20pt}
\TabPositions{\bibmargin}
\newcommand{\quotes}[1]{``#1''}
\newcommand{\insertbibliography}[1]{%
  \setlength{\parindent}{0pt}
  %\setlength{\bibsep}{4pt}
  %\setlength\itemindent{0pt}
  %\setlength{\bibhang}{\bibmargin}
  {\setstretch{0.85}
    \fontsize{8}{10} \selectfont
    \bibliographystyle{un}
    \renewcommand\refname{References}
    \bibliography{#1}}
}

% Definitions for capital letter
\newcommand{\firstword}[2]{\lettrine[findent=2pt]{\textbf{#1}}{}#2 }

\if@isblind
  \makeatletter
  \AtBeginDocument{
    \hypersetup{
      pdfinfo={
          Title={\@title},
          Author={AUTHOR},
          Keywords={\@englishkeywords},
          Producer={AJCEAM/UFT},
          Subject={\@englishabstract}
        }
    }
  }
  \makeatother
\else
  \makeatletter
  \AtBeginDocument{
    \hypersetup{
      pdfinfo={
          Title={\@title},
          Author={\@thefirstauthor},
          Keywords={\@englishkeywords},
          Producer={AJCEAM/UFT},
          Subject={\@englishabstract}
        }
    }
  }
  \makeatother
\fi
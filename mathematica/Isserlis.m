BeginPackage["CommitteeIsserlis`"]
  IsserlisTheorem::usage = 
    "Compute the expectation of a polynomial over a multivariate gaussian."
  RawMoment::usage = 
    "Compute the moment of a multivariate Gaussian."
  IsserlisCovariance::usage = 
    "Compute the covariance of a polynomial over a multivariate gaussian."

 LocalFieldNumber = 6;

  (*The extra field is the artificial noise*)
  TotalDimension =LocalFieldNumber+1;

  lf = \[Lambda]


  (*Extended Local Fields*)
  LocalFields = Range[TotalDimension];
  For[it = 1, it <=LocalFieldNumber, it++, LocalFields[[it]] = lf[it] ]; (* the iterator is not a local variable! Be careful!!*)
  LocalFields[[TotalDimension]] = \[Xi];


  (*Extended Covariance Matrix for local field and artificial noise*)
  Omega = Normal@SymmetrizedArray[{i_, j_} -> omega[i, j], {TotalDimension, TotalDimension}, Symmetric[{1, 2}] ] /. {
        omega[i_, j_] :> If[i <=LocalFieldNumber,
            If[j <=LocalFieldNumber,\[Omega][i,j], 0],
            If[j <=LocalFieldNumber,0, \[CapitalDelta]]
          ]
      };
  
  (*The moment generating function for local fields*)
  LFMomentGeneratingFunction = Exp[1/2* LocalFields.Omega.LocalFields];
  Begin["Private`"]
  

  RawMoment[exponents__] := 
    Module[{derivativespairs, result},
      derivativepairs = Transpose@{LocalFields, List[exponents]};
      result = D[LFMomentGeneratingFunction, ##] & @@ derivativepairs;
      result /. {
        lf[i_] :> 0,
        \[Xi] :> 0
      }
    ];

  IsserlisTheorem[polynomial_] :=
    (* CoefficientRules[polynomial, LocalFields]; *)
    ExpandAll[Total[Replace[
      CoefficientRules[polynomial, LocalFields],
      {Rule -> Times, List -> RawMoment}, {2, 3}, Heads -> True
    ] ] ];

  IsserlisCovariance[polynomial1_, polynomial2_] :=
    Module[{polynomial, result},
      polynomial = polynomial1*polynomial2;
      result = IsserlisTheorem[polynomial];
      result - IsserlisTheorem[polynomial1]*IsserlisTheorem[polynomial2]
    ];
  
    
  End[]
EndPackage[]
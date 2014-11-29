lbfgs-cuda
==========


In Optimization Science, the algorithm of Limited Broyden-Fletcher-Goldfarb-Shanno is an interactive method for solving unconstrained nonlinear optimization problems; its source code is open.  

This paper analyses and then implements the parallel version of this algorithm accelerating it by using GPU with CUDA technology. 

For the task of writing parallel code, it was opted to use cuBLAS library. Thus the effort of software development is mitigated, and the final code is obtained with a high degree of maintainability, for the source code is not written 
using the highly complex native CUDA code.

The work was validated with computational experiments. The experimental results are analyzed,
and it is determined some conditions for the occurrence of speedup.




Dentro da linha de Otimização o algoritmo de Limited Broyden-Fletcher-Goldfarb-Shanno é um método interativo para resolução de problemas irrestritos de otimização não-linear; seu código fonte é aberto.

Este trabalho analisa e em seguida implementa a versão paralelizada desse algoritmo acelerando-o com o uso de GPU e tecnologia CUDA.

Para o trabalho de escrever o código paralelo, optou-se pelo uso da biblioteca cuBLAS. Dessa forma o esforço de desenvolvimento de software é mitigado, e obtêm-se um código final com alto grau de manutenibilidade, pois desenvolve-se software, descartando a alta complexidade de escrita em código nativo CUDA.

O trabalho foi validado com experimentos computacionais. Os resultados experimentais são analisados, com isso determinam-se algumas condições para ocorrência de {\em speedup}.


by Eduardo Bomfim Sanseverino

Oriented by: Sergio Barbosa Villas-Boas (sbVB)
www.sbvb.com.br
sbvillasboas@gmail.com




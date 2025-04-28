# Este script é executado no nó de login. 
#
# O sbatch envia um script para ser executado em um nó.
#
# "-A" indica a conta do nosso projeto;
# "-p" indica a partição em que o nó está;
# "-o" indica o arquivo de saída para stdout e stderr;
#
# O run_mig.sh deve ser substituido por outro script chamando seu programa.
# Os argumentos após ele na linha abaixo serão passados para ele. 
# Neste caso, serão o primeiro e segundo argumento deste script.
# O terceiro indica o caminho relativo do arquivo de saída.s

WORK_DIR="/home/joao.santana/visco_cluster/code_rtm_implemnte/code_with_matplot/kv2/logs/log.txt"

sbatch -A geo-inct -p standard -o $WORK_DIR run_mig.sh $1 $2

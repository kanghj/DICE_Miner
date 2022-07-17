apt-get update && apt-get install -y python3 python3-pip graphviz maven git
pip3 install graphviz


cd /workspace/DICE_Tester 
git config --global --add safe.directory /workspace/DICE_Tester
git checkout DICE
mvn clean package -DskipTests
cd ../DICE_dummy_code/
cp ../DICE_Tester/master/target/evosuite-master-1.0.7-DICE.jar .


# setup (we run the test generator to use its static analyzer to obtain the pure methods)
# an alternative is to use ReImInfer from "Wei Huang et al. 2012. ReImInfer: method purity inference for Java." (done in the original experiments)
# but this way is easier
mkdir evosuite-tests
mvn clean compile -DskipTests

# the following commands will end in errors, but they are expected.
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.ArrayList -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.LinkedList -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.HashSet -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.HashMap -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.Hashtable -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.zip.ZipOutputStream -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.security.Signature -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.net.Socket -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.NumberFormatStringTokenizer -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.StackAr -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.StringTokenizer -projectCP target/classes/ -Dsearch_budget=10
timeout 30s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.zip.ZipOutputStream -projectCP target/classes/ -Dsearch_budget=10

cd ../DICE_Miner

python3 ltl_rules.py original_traces/ArrayList/input.txt ../DICE_dummy_code/evosuite-tests/ArrayList.pures arraylist.vocab.txt | grep "LTL" > ltl_arraylist.txt
python3 ltl_rules.py original_traces/LinkedList/input.txt ../DICE_dummy_code/evosuite-tests/LinkedList.pures linkedlist.vocab.txt | grep "LTL" > ltl_linkedlist.txt
python3 ltl_rules.py original_traces/HashSet/input.txt ../DICE_dummy_code/evosuite-tests/HashSet.pures hashset.vocab.txt | grep "LTL" > ltl_hashset.txt
python3 ltl_rules.py original_traces/HashMap/input.txt ../DICE_dummy_code/evosuite-tests/HashMap.pures hashmap.vocab.txt | grep "LTL" > ltl_hashmap.txt
python3 ltl_rules.py original_traces/Hashtable/input.txt ../DICE_dummy_code/evosuite-tests/Hashtable.pures hashtable.vocab.txt | grep "LTL" > ltl_hashtable.txt
python3 ltl_rules.py original_traces/ZipOutputStream/input.txt ../DICE_dummy_code/evosuite-tests/ZipOutputStream.pures zipoutputstream.vocab.txt | grep "LTL" > ltl_zipoutputstream.txt
python3 ltl_rules.py original_traces/Signature/input.txt ../DICE_dummy_code/evosuite-tests/Signature.pures signature.vocab.txt | grep "LTL" > ltl_signature.txt
python3 ltl_rules.py original_traces/Socket/input.txt ../DICE_dummy_code/evosuite-tests/Socket.pures socket.vocab.txt | grep "LTL" > ltl_socket.txt
python3 ltl_rules.py original_traces/NumberFormatStringTokenizer/input.txt ../DICE_dummy_code/evosuite-tests/NumberFormatStringTokenizer.pures numberformatstringtokenizer.vocab.txt | grep "LTL" > ltl_numberformatstringtokenizer.txt
python3 ltl_rules.py original_traces/StackAr/input.txt ../DICE_dummy_code/evosuite-tests/StackAr.pures stackar.vocab.txt | grep "LTL" > ltl_stackar.txt
python3 ltl_rules.py original_traces/StringTokenizer/input.txt ../DICE_dummy_code/evosuite-tests/StringTokenizer.pures stringtokenizer.vocab.txt | grep "LTL" > ltl_stringtokenizer.txt



mkdir dice_tester_traces

mv *.vocab.txt ../DICE_dummy_code
mv ltl_*.txt ../DICE_dummy_code


cd ../DICE_dummy_code

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_arraylist.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.ArrayList -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.util.ArrayList.traces ../DICE_Miner/dice_tester_traces/ArrayList_enhanced.traces

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_linkedlist.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.LinkedList -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.util.LinkedList.traces ../DICE_Miner/dice_tester_traces/LinkedList_enhanced.traces

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_hashset.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.HashSet -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.util.HashSet.traces ../DICE_Miner/dice_tester_traces/HashSet_enhanced.traces

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_hashmap.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.HashMap -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.util.HashMap.traces ../DICE_Miner/dice_tester_traces/HashMap_enhanced.traces

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_hashtable.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.Hashtable -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.util.Hashtable.traces ../DICE_Miner/dice_tester_traces/Hashtable_enhanced.traces

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_signature.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.security.Signature -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.security.Signature.traces ../DICE_Miner/dice_tester_traces/Signature_enhanced.traces

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_socket.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.net.Socket -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.net.Socket.traces ../DICE_Miner/dice_tester_traces/Socket_enhanced.traces

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_numberformatstringtokenizer.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.NumberFormatStringTokenizer -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.NumberFormatStringTokenizer.traces ../DICE_Miner/dice_tester_traces/NumberFormatStringTokenizer_enhanced.traces

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_stackar.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.StackAr -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.StackAr.traces ../DICE_Miner/dice_tester_traces/StackAr_enhanced.traces

rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_stringtokenizer.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.StringTokenizer -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.util.StringTokenizer.traces ../DICE_Miner/dice_tester_traces/StringTokenizer_enhanced.traces


rm null*traces

echo "path:/workspace/DICE_dummy_code/ltl_zipoutputstream.txt" > config.txt 
timeout 1000s java -jar evosuite-master-1.0.7-DICE.jar -class tutorial.util.zip.ZipOutputStream -projectCP target/classes/ -Dsearch_budget=900
mv null_tutorial.util.zip.ZipOutputStream.traces ../DICE_Miner/dice_tester_traces/ZipOutputStream_enhanced.traces




cd ../DICE_Miner
mkdir tmp_outputs

sed -i 's/<init>/ArrayList/g' dice_tester_traces/ArrayList_enhanced.traces
sed -i 's/<init>/LinkedList/g' dice_tester_traces/LinkedList_enhanced.traces
sed -i 's/<init>/HashSet/g' dice_tester_traces/HashSet_enhanced.traces
sed -i 's/<init>/HashMap/g' dice_tester_traces/HashMap_enhanced.traces
sed -i 's/<init>/Hashtable/g' dice_tester_traces/Hashtable_enhanced.traces
sed -i 's/<init>/Signature/g' dice_tester_traces/Signature_enhanced.traces
sed -i 's/<init>/Socket/g' dice_tester_traces/Socket_enhanced.traces
sed -i 's/<init>/NumberFormatStringTokenizer/g' dice_tester_traces/NumberFormatStringTokenizer_enhanced.traces
sed -i 's/<init>/StackAr/g' dice_tester_traces/StackAr_enhanced.traces
sed -i 's/<init>/StringTokenizer/g' dice_tester_traces/StringTokenizer_enhanced.traces
sed -i 's/<init>/ZipOutputStream/g' dice_tester_traces/ZipOutputStream_enhanced.traces
grep -v "IO-LEAK" dice_tester_traces/ZipOutputStream_enhanced.traces > dice_tester_traces/tmp_file 
mv dice_tester_traces/tmp_file dice_tester_traces/ZipOutputStream_enhanced.traces

python3 main.py original_traces/ArrayList/input.txt ../DICE_dummy_code/evosuite-tests/ArrayList.pures  dice_tester_traces/ArrayList_enhanced.traces   tmp_outputs/ArrayList/fsm.txt 3000 
python3 main.py original_traces/LinkedList/input.txt ../DICE_dummy_code/evosuite-tests/LinkedList.pures  dice_tester_traces/LinkedList_enhanced.traces   tmp_outputs/LinkedList/fsm.txt 3000 
python3 main.py original_traces/HashSet/input.txt ../DICE_dummy_code/evosuite-tests/HashSet.pures  dice_tester_traces/HashSet_enhanced.traces   tmp_outputs/HashSet/fsm.txt 3000 
python3 main.py original_traces/HashMap/input.txt ../DICE_dummy_code/evosuite-tests/HashMap.pures  dice_tester_traces/HashMap_enhanced.traces   tmp_outputs/HashMap/fsm.txt 3000
python3 main.py original_traces/Hashtable/input.txt ../DICE_dummy_code/evosuite-tests/Hashtable.pures  dice_tester_traces/Hashtable_enhanced.traces   tmp_outputs/Hashtable/fsm.txt 3000  
python3 main.py original_traces/Signature/input.txt ../DICE_dummy_code/evosuite-tests/Signature.pures  dice_tester_traces/Signature_enhanced.traces   tmp_outputs/Signature/fsm.txt 300 
python3 main.py original_traces/Socket/input.txt ../DICE_dummy_code/evosuite-tests/Socket.pures  dice_tester_traces/Socket_enhanced.traces   tmp_outputs/Socket/fsm.txt 300 
python3 main.py original_traces/NumberFormatStringTokenizer/input.txt ../DICE_dummy_code/evosuite-tests/NumberFormatStringTokenizer.pures  dice_tester_traces/NumberFormatStringTokenizer_enhanced.traces   tmp_outputs/NumberFormatStringTokenizer/fsm.txt 300
python3 main.py original_traces/StackAr/input.txt ../DICE_dummy_code/evosuite-tests/StackAr.pures  dice_tester_traces/StackAr_enhanced.traces   tmp_outputs/StackAr/fsm.txt 300 
python3 main.py original_traces/StringTokenizer/input.txt ../DICE_dummy_code/evosuite-tests/StringTokenizer.pures  dice_tester_traces/StringTokenizer_enhanced.traces   tmp_outputs/StringTokenizer/fsm.txt 300 
python3 main.py original_traces/ZipOutputStream/input.txt ../DICE_dummy_code/evosuite-tests/ZipOutputStream.pures  dice_tester_traces/ZipOutputStream_enhanced.traces   tmp_outputs/ZipOutputStream/fsm.txt 300



python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/ArrayList/ --ground_truth_folder /workspace/fsa_model_ground_truths/ArrayList/ --result_folder /DICE_eval_results/ArrayList --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/LinkedList/ --ground_truth_folder /workspace/fsa_model_ground_truths/LinkedList/ --result_folder /DICE_eval_results/LinkedList --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/HashSet/ --ground_truth_folder /workspace/fsa_model_ground_truths/HashSet/ --result_folder /DICE_eval_results/HashSet --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/HashMap/ --ground_truth_folder /workspace/fsa_model_ground_truths/HashMap/ --result_folder /DICE_eval_results/HashMap --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/Hashtable/ --ground_truth_folder /workspace/fsa_model_ground_truths/Hashtable/ --result_folder /DICE_eval_results/Hashtable --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/Signature/ --ground_truth_folder /workspace/fsa_model_ground_truths/signature/ --result_folder /DICE_eval_results/Signature --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/Socket/ --ground_truth_folder /workspace/fsa_model_ground_truths/socket/ --result_folder /DICE_eval_results/Socket --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/NumberFormatStringTokenizer/ --ground_truth_folder /workspace/fsa_model_ground_truths/NumberFormatStringTokenizer/ --result_folder /DICE_eval_results/numberformatstringtokenizer --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/StackAr/ --ground_truth_folder /workspace/fsa_model_ground_truths/stackar/ --result_folder /DICE_eval_results/StackAr --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/StringTokenizer/ --ground_truth_folder /workspace/fsa_model_ground_truths/stringtokenizer/ --result_folder /DICE_eval_results/StringTokenizer --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000
python3 /workspace/deep_spec_learning_ws/deep_spec_learning/model_learning/evaluation/evaluate_clustered_automata.py --cluster_folder /workspace/DICE_Miner/tmp_outputs/ZipOutputStream/ --ground_truth_folder /workspace/fsa_model_ground_truths/zipoutputstream/ --result_folder /DICE_eval_results/ZipOutputStream --ignore_method_suffix 1 --overall_min_label_coverage 20 --max_label_repeated_per_trace 3 --max_trace_length 50 --max_num_trace 10000

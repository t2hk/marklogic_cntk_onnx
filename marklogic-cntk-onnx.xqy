xquery version "1.0-ml";

(:推論データの読み込み:)
let $x_test :=
  for $x in xdmp:directory("/iris/test/", "infinity")
    let $test := fn:concat($x/iris/sepal_length/text(),",", $x/iris/sepal_width/text(), ",", $x/iris/petal_length/text(), ",", $x/iris/petal_width/text())
    let $label_text := $x/iris/species/text()
    let $label := 
      if ($label_text = "setosa") then 1
      else if ($label_text = "versicolor") then 2
      else if ($label_text = "virginica") then 3
      else -1
    
    return json:to-array((($x/iris/sepal_length/text(), $x/iris/sepal_width/text(), $x/iris/petal_length/text(),  $x/iris/petal_width/text(), $label)))

(:ONNXモデルの読み込む。:)
let $doc := fn:doc("/model/cntk_iris_model.onnx")/binary()
let $model := cntk:function($doc, cntk:gpu(0), "onnx")

(:モデルの入出力形式を取得する。:)
let $input_variable := cntk:function-arguments($model)
let $output_variable := cntk:function-output($model)

(:テストデータに対して推論する。:)
for $value at $i in $x_test
  (:テスト用のデータと正解ラベルを作成する。:)
  let $test := json:to-array(($value[1], $value[2], $value[3], $value[4]))
  let $input_value := cntk:batch(cntk:variable-shape($input_variable), json:to-array(($test)))
  let $input_pair := json:to-array(($input_variable, $input_value))
  let $label := $value[5]
  
  (:ONNXモデルを使用して推論する。:)
  let $output_value := cntk:evaluate($model, $input_pair, $output_variable)

  (:結果を出力する。:)
  let $infer_result := cntk:value-to-array($output_variable, $output_value)
  return
  for $res at $i in $infer_result
    return fn:concat("iris:",$test, " collect:", $label, " infer:",fn:index-of($res[1], fn:max($res[1])))


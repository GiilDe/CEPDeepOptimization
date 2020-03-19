import com.espertech.esper.common.client.EPCompiled;
import com.espertech.esper.common.client.EventBean;
import com.espertech.esper.common.client.configuration.Configuration;
import com.espertech.esper.common.internal.event.bean.core.BeanEventBean;
import com.espertech.esper.common.internal.event.map.MapEventBean;
import com.espertech.esper.compiler.client.CompilerArguments;
import com.espertech.esper.compiler.client.EPCompileException;
import com.espertech.esper.compiler.client.EPCompiler;
import com.espertech.esper.compiler.client.EPCompilerProvider;
import com.espertech.esper.runtime.client.*;

import java.io.*;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;


public class Main {
    final static String scoresPath = "Data/scores.txt";

    final static String trainStreamPath = "Data/train data stream.txt";
    final static String trainOutputPath = "Application/output data/train output.txt";
    final static String trainElaboratedOutputPath = "Application/output data/train elaborated output.txt";
    final static int trainEventsNum = 10000209;

    final static String testStreamPath = "Data/test data stream.txt";
    final static String testOutputPath = "Application/output data/test output.txt";
    final static String testElaboratedOutputPath = "Application/output data/test elaborated output.txt";
    final static int testEventsNum = 1000000;

    final static String testFilterOutputPath = "Application/output data/test filter output.txt";
    final static String testFilterElaboratedOutputPath = "Application/output data/test filter elaborated output.txt";

    final static Map<Integer, List<List<Integer>>> eventToOccurrences = new HashMap<>();

    final static String statementName = "mystatement";
    final static int toPrint = 1000;
    final static boolean shouldPrint = true;
    final static String patternPath = "Application/pattern";
    static int count = 0;

    public static void main(String[] s) throws IOException {
//        query(trainStreamPath, trainOutputPath, trainElaboratedOutputPath, null, trainEventsNum);
        //query(testStreamPath, testOutputPath, testElaboratedOutputPath, null, testEventsNum);
        query(testStreamPath, testFilterOutputPath, testFilterElaboratedOutputPath, scoresPath, testEventsNum);
    }

    private static void query(String inputPath, String outputPath, String elaboratedOutputPath, String scoresPath, int eventsNum)
            throws FileNotFoundException {
        for (int count = 0; count < eventsNum; count++) {
            eventToOccurrences.put(count, new LinkedList<>());
        }

        EPCompiler compiler = EPCompilerProvider.getCompiler();
        Configuration configuration = new Configuration();
        configuration.getCommon().addEventType(Event.class);
        CompilerArguments args = new CompilerArguments(configuration);

        EPCompiled epCompiled;
        String pattern;
        try {
            pattern = getPattern();
            epCompiled = compiler.compile("@name('" + statementName + "') " + pattern, args);
        }
        catch (EPCompileException ex) {
            // handle exception here
            throw new RuntimeException(ex);
        }

        EPRuntime runtime = EPRuntimeProvider.getDefaultRuntime(configuration);
        EPDeployment deployment;
        try {
            deployment = runtime.getDeploymentService().deploy(epCompiled);
        }
        catch (EPDeployException ex) {
            // handle exception here
            throw new RuntimeException(ex);
        }

        EPStatement statement =
                runtime.getDeploymentService().getStatement(deployment.getDeploymentId(), statementName);

        statement.addListener((newData, oldData, s, r) -> {
            for (EventBean eventBean : newData) {
                Map<String, Object> events = ((MapEventBean) eventBean).getProperties();
                List<Integer> counts = new LinkedList<>();
                for (Object b : events.values()) {
                    Event event;
                    if(b instanceof BeanEventBean){
                        event = (Event) ((BeanEventBean) b).getUnderlying();
                    }
                    else {
                        event = (Event) b;
                    }
                    counts.add(event.getCount());
                }
                for (Object b : events.values()) {
                    Event event;
                    if(b instanceof BeanEventBean){
                        event = (Event) ((BeanEventBean) b).getUnderlying();
                    }
                    else {
                        event = (Event) b;
                    }
                    List<List<Integer>> l = eventToOccurrences.get(event.getCount());
                    l.add(counts);
                }
            }
        });

        sendEvents(runtime, inputPath, scoresPath);

        PrintWriter writer = new PrintWriter(outputPath);
        PrintWriter elaboratedWriter = new PrintWriter(elaboratedOutputPath);

        for (int count = 0; count < eventsNum; count++) {
            List<List<Integer>> matches = eventToOccurrences.get(count);
            writer.println(matches.size());
            elaboratedWriter.print(matches.size());
            for(List<Integer> match : matches){
                elaboratedWriter.print("; ");
                for(int eventCount : match){
                    elaboratedWriter.print(eventCount + ", ");
                }
            }
            elaboratedWriter.println("");
        }
        writer.close();
        elaboratedWriter.close();
        eventToOccurrences.clear();
        count = 0;
    }

    private static String getPattern() {
        StringBuilder pattern = new StringBuilder();
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(patternPath));
            String firstLine = reader.readLine();
            pattern.append(firstLine);
            String line = reader.readLine();
            while (line != null && !line.contentEquals("done")) {
                // read next line
                pattern.append(" ").append(line);
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return pattern.toString();
    }

    static boolean eventFilter(float score){
        return score > 0.1653 || score == -1;
    }

    static void sendEvents(EPRuntime runtime, String inputPath, String scoresPath){
        BufferedReader reader;
        BufferedReader scoresReader = null;
        int done = -1;
        int notSent = 0;
        try {
            reader = new BufferedReader(new FileReader(inputPath));
            if(scoresPath != null){
                scoresReader = new BufferedReader(new FileReader(scoresPath));
            }
            String line = reader.readLine();
            String score = null;
            if(scoresReader != null) {
                score = scoresReader.readLine();
            }
            while (line != null && count != done) {
                String[] s = line.split(",");
                Event event;
                if(score == null || eventFilter(Float.parseFloat(score))) {
                    event = new Event(s[0], Double.parseDouble(s[1]), count);
                }
                else{
                    notSent += 1;
                    event = new Event("-1", -1, count);
                }
                runtime.getEventService().sendEventBean(event, "Event");
                line = reader.readLine();
                if(scoresReader != null) {
                    score = scoresReader.readLine();
                }
                count += 1;
                if(shouldPrint && count % toPrint == 0){
                    System.out.println(count);
                }
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

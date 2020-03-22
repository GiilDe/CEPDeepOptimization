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
import org.antlr.v4.runtime.misc.Pair;

import java.io.*;
import java.util.*;


public class Main {
    final static String scoresPath = "../Data/scores.txt";

    final static String trainStreamPath = "../Data/train data stream.txt";
    final static String trainMatchesPath = "output data/train matches2.txt";
    final static String trainElaboratedMatchesPath = "output data/train elaborated matches2.txt";
    final static int trainEventsNum = 10000209;

    final static String testStreamPath = "../Data/test data stream.txt";
    final static String testMatchesPath = "output data/test matches.txt";
    final static String testElaboratedMatchesPath = "output data/test elaborated matches.txt";
    final static int testEventsNum = 1000000;

    final static String testFilterMatchesPathPrefix = "output data/test matches";
    final static String testFilterElaboratedMatchesPathPrefix = "output data/test elaborated matches";

    final static String testFilterDetailsPathPrefix = "output data/output details/details";


    final static Map<Integer, List<List<Integer>> > idToMatches = new HashMap<>();
    final static Set< Set<Integer> > matches = new HashSet<>();

    final static String statementName = "mystatement";
    final static int toPrint = 1000;
    final static boolean printProgress = true;
    final static String patternPath = "pattern";
    final static int windowSize = 8;
    static int matchesNum = 0;


    public static UpdateListener listener = (newData, oldData, s, r) -> {
        for (EventBean match : newData) {
            Map<String, Object> events = ((MapEventBean) match).getProperties();
            List<Integer> matchCounts = getMatch(events);
            final HashSet<Integer> matchCountsSet = new HashSet<>(matchCounts);
            if (!matches.contains(matchCountsSet)) {
                matches.add(matchCountsSet);
                matchesNum += 1;

                for (Object b : events.values()) {
                    Event event;
                    if (b instanceof BeanEventBean) {
                        event = (Event) ((BeanEventBean) b).getUnderlying();
                    } else {
                        event = (Event) b;
                    }
                    List<List<Integer>> l = idToMatches.get(event.getId());
                    l.add(matchCounts);
                }
            }
            else {
                int x = 0;
            }
        }
    };


    public static void main(String[] s) throws IOException {
        query(testStreamPath, testFilterMatchesPathPrefix, testFilterElaboratedMatchesPathPrefix,
                scoresPath, testEventsNum, 0.1, 4);
    }

    private static void queryTrainData() throws FileNotFoundException {
        query(trainStreamPath, trainMatchesPath, trainElaboratedMatchesPath, null, trainEventsNum,
                null, null);
    }

    private static void testThresholds() throws FileNotFoundException {
        double[] thresholds = new double[]{0, 0.25, 0.5, 0.75, 1, 1.25, 1.5};
        for(double threshold : thresholds){
            query(testStreamPath, testFilterMatchesPathPrefix, testFilterElaboratedMatchesPathPrefix,
                    scoresPath, testEventsNum, threshold, null);
        }
    }

    private static void testMaximumK() throws FileNotFoundException {
        int[] ks = new int[]{3, 4, 5, 6, 7, 8};
        //for(int k : ks){
        query(testStreamPath, testFilterMatchesPathPrefix, testFilterElaboratedMatchesPathPrefix,
                scoresPath, testEventsNum, 0.0, 7);
        //}
    }

    private static void query(String inputPath, String outputPath, String elaboratedOutputPath, String scoresPath,
                              int eventsNum, Double threshold, Integer k)
            throws FileNotFoundException {
        for (int count = 0; count < eventsNum; count++) {
            idToMatches.put(count, new LinkedList<>());
        }

        if(threshold == null){
            threshold = (double) 0;
        }

        EPRuntime runtime = getEpRuntime();

        int sentEventsNum = 0;
        try {
            if(k != null){
                sentEventsNum = sendEventsMaximumK(runtime, inputPath, scoresPath, threshold, k);
            }
            else {
                sentEventsNum = sendEvents(runtime, inputPath, scoresPath, threshold);
            }
        }catch (IOException e){
            e.printStackTrace();
        }

        String kString = "_";
        String label = Double.toString(threshold);
        if(k != null){
            kString = "_k_" + (double) k;
            label = Integer.toString(k);
        }

        String postfix  = "_threshold_" + threshold + kString + "_.txt";
        outputPath = outputPath + postfix;
        elaboratedOutputPath = elaboratedOutputPath + postfix;
        String filterDetailsPath = testFilterDetailsPathPrefix + postfix;
        PrintWriter filterDetails = new PrintWriter(filterDetailsPath);
        filterDetails.println(label);
        filterDetails.println(sentEventsNum);
        filterDetails.println(matchesNum);
        filterDetails.close();

        PrintWriter writer = new PrintWriter(outputPath);
        PrintWriter elaboratedWriter = new PrintWriter(elaboratedOutputPath);

        for (int count = 0; count < eventsNum; count++) {
            List<List<Integer>> matches = idToMatches.get(count);
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
        idToMatches.clear();
        matches.clear();
        matchesNum = 0;
        runtime.destroy();
        System.out.println("Done");
    }

    private static EPRuntime getEpRuntime() {
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

        statement.addListener(listener);
        return runtime;
    }

    private static List<Integer> getMatch(Map<String, Object> events) {
        List<Integer> counts = new LinkedList<>();
        for (Object b : events.values()) {
            Event event;
            if(b instanceof BeanEventBean){
                event = (Event) ((BeanEventBean) b).getUnderlying();
            }
            else {
                event = (Event) b;
            }
            counts.add(event.getId());
        }
        return counts;
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

    static boolean eventFilter(float score, double threshold){
        return score >= threshold || score == -1;
    }

    static int sendEvents(EPRuntime runtime, String inputPath, String scoresPath, double threshold) throws IOException {
        BufferedReader reader;
        BufferedReader scoresReader = null;
        int notSent = 0;
        int count = 0;
        Event dummyEvent = new Event("-1", -1, -1, -1);
        reader = new BufferedReader(new FileReader(inputPath));

        if(scoresPath != null){
            scoresReader = new BufferedReader(new FileReader(scoresPath));
        }
        String score = null;
        if(scoresReader != null) {
            score = scoresReader.readLine();
        }

        Event event = getNextEvent(reader, count, count);
        while (event != null) {

            Event eventForUse;
            if(score == null || eventFilter(Float.parseFloat(score), threshold)) {
                eventForUse = event;
            }
            else{
                notSent += 1;
                eventForUse = dummyEvent;
            }
            runtime.getEventService().sendEventBean(eventForUse, "Event");

            count += 1;
            if(printProgress && (count % toPrint == 0)){
                System.out.println(count);
            }

            if(scoresReader != null) {
                score = scoresReader.readLine();
            }
            event = getNextEvent(reader, count, count);
        }
        reader.close();
        if(scoresReader != null){
            scoresReader.close();
        }
        return testEventsNum - notSent;
    }

    static int sendEventsMaximumK(EPRuntime runtime, String inputPath, String scoresPath, double threshold, int k)
            throws IOException{
        BufferedReader reader;
        BufferedReader scoresReader;

        Event dummyEvent = new Event("-1", -1, -1, -1);
        List<Pair<Event, Float>> window = new LinkedList<>();
        reader = new BufferedReader(new FileReader(inputPath));
        scoresReader = new BufferedReader(new FileReader(scoresPath));

        int count = 0;
        int id = 0;
        for (int i = 0; i < windowSize; i++) {
            Event event = getNextEvent(reader, count, id);
            String score = scoresReader.readLine();
            window.add(new Pair<>(event, Float.parseFloat(score)));

            count += 1;
            id += 1;
        }

        boolean done = false;
        while (!done) {
            window.sort(Comparator.comparing(p -> p.b));
            Collections.reverse(window);

            var iterator = window.listIterator();
            List<Pair<Event, Float>> topK = new LinkedList<>();
            for (int j = 0; j < k; j++) {
                final var next = iterator.next();
                topK.add(next);
            }

            topK.sort(Comparator.comparing(p -> p.a.getCount()));
            var iterator2 = topK.listIterator();

            for (int j = 0; j < k; j++) {
                final var next = iterator2.next();
                Event event = next.a;
                Float score = next.b;
                Event eventForUse;
                if(eventFilter(score, threshold)) {
                    eventForUse = event;
                }
                else {
                    eventForUse = dummyEvent;
                }
                runtime.getEventService().sendEventBean(eventForUse, "Event");
            }

            for (int i = 0; i < windowSize; i++) {
                runtime.getEventService().sendEventBean(dummyEvent, "Event");
            }

            window.sort(Comparator.comparing(p -> p.a.getCount()));
            window.remove(0);

            Event event = getNextEvent(reader, count, id);
            String score = scoresReader.readLine();

            if (event == null) {
                done = true;
            }
            else {
                window.add(new Pair<>(event, Float.parseFloat(score)));
            }


            count += 2*windowSize;
            for (var p : window) {
                Event e = p.a;
                e.setCount(e.getCount() + 2*windowSize - 1);
            }

            id += 1;
            if(printProgress && (id % toPrint == 0)){
                System.out.println(id);
            }
        }

        reader.close();
        scoresReader.close();
        return testEventsNum;
    }

    private static Event getNextEvent(BufferedReader reader, int count, int id) throws IOException {
        String line = reader.readLine();
        if(line != null) {
            String[] s = line.split(",");
            return new Event(s[0], Double.parseDouble(s[1]), count, id);
        }
        else{
            return null;
        }
    }
}

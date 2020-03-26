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
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.*;


public class Main {
    final static String scoresPath = "Data/scores.txt";

    final static String trainStreamPath = "Data/train data stream.txt";
    final static String trainMatchesPath = "Application/output data/matches/train matches.txt";
    final static String trainElaboratedMatchesPath = "Application/output data/matches/train elaborated matches.txt";
    final static int trainEventsNum = 7913670;

    final static String testStreamPath = "Data/test data stream.txt";
    final static String testMatchesPath = "Application/output data/matches/test matches.txt";
    final static String testElaboratedMatchesPath = "Application/output data/matches/test elaborated matches.txt";
    final static int testEventsNum = 1000000;

    final static String testMatchesPathPrefix = "Application/output data/matches/test matches";
    final static String testElaboratedMatchesPathPrefix = "Application/output data/matches/test elaborated matches";

    final static String testFilterDetailsPathPrefix = "Application/output data/details/threshold details/details";

    final static String patternPath = "Application/pattern";

    final static Map<Integer, List<List<Integer>> > idToMatches = new HashMap<>();
    final static Set< Set<Integer> > matches = new HashSet<>();

    final static String statementName = "mystatement";
    final static int toPrint = 1000;
    final static boolean printProgress = true;
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
                System.err.println("Bad");
                System.out.println("Bad");
                System.out.println("Bad");
                System.out.println("Bad");
                System.out.println("Bad");
                System.out.println("Bad");
            }
        }
    };


    public static void main(String[] s) {
        try {
            query(testStreamPath, testMatchesPathPrefix, testElaboratedMatchesPathPrefix, testFilterDetailsPathPrefix,
                    scoresPath, testEventsNum, 0.2, null, true);
        } catch (IOException ex){
            ex.printStackTrace();
        }
    }

    private static void queryTrainData() throws IOException {
        query(trainStreamPath, trainMatchesPath, trainElaboratedMatchesPath, testFilterDetailsPathPrefix,
                null, trainEventsNum, null, null, true);
    }

    private static void testThresholds() throws IOException {
        double threshold = 0.0;
        for (int i = 0; i < 10; i++) {
            query(testStreamPath, testMatchesPathPrefix, testElaboratedMatchesPathPrefix, testFilterDetailsPathPrefix,
                    scoresPath, testEventsNum, threshold, null, false);
            threshold += 0.12;
        }
    }

    private static void testThresholdsNormalized() throws IOException {
        double[] thresholds = new double[]{0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32};
        for(double threshold : thresholds){
            query(testStreamPath, testMatchesPathPrefix, testElaboratedMatchesPathPrefix,
                    testFilterDetailsPathPrefix, scoresPath, testEventsNum, threshold, null, false);
        }
    }

    private static void testMaximumK() throws IOException {
        int[] ks = new int[]{3, 4, 5, 6, 7, 8};
        //for(int k : ks){
        query(testStreamPath, testMatchesPathPrefix, testElaboratedMatchesPathPrefix, testFilterDetailsPathPrefix,
                scoresPath, testEventsNum, 0.0, 7, false);
        //}
    }

    private static void query(String inputPath, String outputPath, String elaboratedOutputPath, String detailsPath,
                              String scoresPath, int eventsNum, Double threshold, Integer k, boolean printMatches)
            throws IOException {
        for (int count = 0; count < eventsNum; count++) {
            idToMatches.put(count, new LinkedList<>());
        }

        if(threshold == null){
            threshold = (double) 0;
        }

        EPRuntime runtime = getEpRuntime();

        int sentEventsNum = 0;
        if(k != null){
            sentEventsNum = sendEventsMaximumK(runtime, inputPath, scoresPath, threshold, k);
        }
        else {
            sentEventsNum = sendEvents(runtime, inputPath, scoresPath, threshold);
        }


        String kString = "_";
        String label = Double.toString(threshold);
        if(k != null){
            kString = "_k_" + (double) k;
            label = Integer.toString(k);
        }

        String postfix  = "_threshold_" + threshold + kString + "_.txt";
        CSVPrinter detailsPrinter = new CSVPrinter(new FileWriter(detailsPath + postfix), CSVFormat.DEFAULT);
        detailsPrinter.printRecord(label);
        detailsPrinter.printRecord(sentEventsNum);
        detailsPrinter.printRecord(matchesNum);
        detailsPrinter.close();

        if(printMatches) {
            outputPath = outputPath + postfix;
            elaboratedOutputPath = elaboratedOutputPath + postfix;
            CSVPrinter writer = new CSVPrinter(new FileWriter(outputPath), CSVFormat.DEFAULT);
            CSVPrinter elaboratedWriter = new CSVPrinter(new FileWriter(elaboratedOutputPath), CSVFormat.DEFAULT);

            for (int count = 0; count < eventsNum; count++) {
                List<List<Integer>> matches = idToMatches.get(count);
                writer.printRecord(matches.size());
                elaboratedWriter.printRecord(matches.size());

                for (List<Integer> match : matches) {
                    elaboratedWriter.print(match);
                }
            }
            writer.close();
            elaboratedWriter.close();
        }
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
        CSVParser eventsParser = CSVParser.parse(new FileReader(inputPath), CSVFormat.DEFAULT);

        CSVParser scoresParser = null;
        Iterator<CSVRecord> scores = null;

        int notSent = 0;
        int count = 0;
        Event dummyEvent = new Event("-1", -1, -1, -1);

        String score = null;

        if(scoresPath != null){
            scoresParser = CSVParser.parse(new FileReader(scoresPath), CSVFormat.DEFAULT);
            scores = scoresParser.iterator();
            score = scores.next().get(0);
        }


        for(CSVRecord s : eventsParser) {
            Event eventForUse;
            if(score == null || eventFilter(Float.parseFloat(score), threshold)) {
                eventForUse = new Event(s.get(0), Double.parseDouble(s.get(1)), count, count);
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

            if(scoresParser != null) {
                score = scores.next().get(0);
            }
        }
        eventsParser.close();
        if(scoresParser != null){
            scoresParser.close();
        }
        return testEventsNum - notSent;
    }

    static int sendEventsMaximumK(EPRuntime runtime, String inputPath, String scoresPath, double threshold, int k)
            throws IOException{
        CSVParser eventsParser = CSVParser.parse(new FileReader(inputPath), CSVFormat.DEFAULT);
        Iterator<CSVRecord> events = eventsParser.iterator();

        CSVParser scoresParser = CSVParser.parse(new FileReader(scoresPath), CSVFormat.DEFAULT);
        Iterator<CSVRecord> scores = scoresParser.iterator();

        Event dummyEvent = new Event("-1", -1, -1, -1);

        int count = 0;
        int id = 0;

        List<Pair<Event, Float>> window = new LinkedList<>();
        for (int i = 0; i < windowSize; i++) {
            CSVRecord s = events.next();
            Event event = new Event(s.get(0), Double.parseDouble(s.get(1)), count, id);

            String score = scores.next().get(0);
            window.add(new Pair<>(event, Float.parseFloat(score)));

            count += 1;
            id += 1;
        }

        while (events.hasNext()) {
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

            CSVRecord s = events.next();

            String score = scores.next().get(0);

            Event event = new Event(s.get(0), Double.parseDouble(s.get(1)), count, id);
            window.add(new Pair<>(event, Float.parseFloat(score)));

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

        scoresParser.close();
        eventsParser.close();
        return testEventsNum;
    }
}
